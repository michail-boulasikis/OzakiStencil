#include "stencil_common.cuh"
#include "stencil_kernels.cuh"

#include "weight_layout.hpp"

#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/tensor_reduce.hpp>

#include <algorithm>
#include <type_traits>

namespace ozaki::gpu {
namespace kernels {

enum class WeightSource { Registers, Shared, Global };

template <int32_t Side, int32_t M> struct Config2d {
  static constexpr Mma2dWeightLayout Weights{Side, M};
  static constexpr int32_t Halo = Side / 2;
  static constexpr int32_t P = Weights.group_cols();       // outputs per MMA row group
  static constexpr int32_t RowsPerK = 32 / P;              // input rows per K block
  static constexpr int32_t KBlocks = Weights.k_blocks();
  static constexpr int32_t N8 = Weights.n8_per_slice();    // 8-column MMA tiles per group
  static constexpr int32_t OutputCols = 16 * P;
  static constexpr int32_t OutputRows = Side == 15 && M == 9 ? 24 : 32;
  static constexpr int32_t WarpRows = 8;
  static constexpr int32_t Threads = OutputRows / WarpRows * WARP_SIZE;
  // Rows of A loaded at once; each B fragment is reused for all of them.
  static constexpr int32_t RowsPerBatch = Side == 15 ? 1 : 4;
  static constexpr int32_t Batches = WarpRows / RowsPerBatch;
  // Shared-memory row pitch.
  static constexpr int32_t SharedColumns = Side == 15 ? 272 : 192;
  // A TMA box row has at most 256 elements, so 15x15 loads 16-bit elements.
  static constexpr int32_t TmaElementBytes = Side == 15 ? 2 : 1;
  static constexpr bool TileLocalMaps = M != 9;
  static constexpr WeightSource Source =
      Side != 15 ? WeightSource::Registers : (M == 9 ? WeightSource::Global : WeightSource::Shared);
#if OZAKI_CUDA_ARCH == 90
  static constexpr int32_t MinBlocks = Side == 15 ? (M == 1 ? 3 : 2) : 1;
#else
  static constexpr int32_t MinBlocks = Side == 15 && M != 9 ? 2 : 1;
#endif

  static constexpr int32_t L = 1;
  using Pipeline = DiagonalSchedule<M, L>;
  static constexpr int32_t TransactionBytes = (OutputRows + 2 * Halo) * SharedColumns;
  // One spare row: the zero-padded last K block reads one row past the box.
  static constexpr int32_t SlotBytes = static_cast<int32_t>(align_up(TransactionBytes + SharedColumns, 128));
  static constexpr int32_t WeightOffset = Pipeline::Slots * SlotBytes;
  static constexpr int32_t SharedBytes = WeightOffset + (Source == WeightSource::Shared ? Weights.bytes() : 0);

  static_assert(SharedColumns >= OutputCols + P && SharedColumns / TmaElementBytes <= 256);
  static_assert(KBlocks * RowsPerK == Side + 1);
};

template <int32_t Side, int32_t M>
__global__ __launch_bounds__(Config2d<Side, M>::Threads, Config2d<Side, M>::MinBlocks) void stencil_2d(
    SlicedTensorView output, PackedStencilArg weight, const __grid_constant__ SlicedTensorTmaView input) {
#pragma nv_diag_suppress static_var_with_dynamic_init
  using C = Config2d<Side, M>;
  using namespace cute;
  constexpr int32_t P = C::P;
  constexpr int32_t SC = C::SharedColumns;
  constexpr Mma2dWeightLayout layout = C::Weights;

  __shared__ __align__(8) uint64_t barriers[C::Pipeline::Slots];
  extern __shared__ __align__(128) core::EmulationType shared[];

  const Schedule<M> schedule(input, output, weight);
  if (schedule.overflowSlices < 0) {
    return;
  }
  typename C::Pipeline pipeline{schedule};
  if (pipeline.nothing_to_do()) {
    return;
  }

  // Start loading input planes.
  const int32_t tileColumn = C::TileLocalMaps ? 0 : static_cast<int32_t>(blockIdx.x) * C::OutputCols / C::TmaElementBytes;
  const int32_t tileRow = static_cast<int32_t>(blockIdx.y) * C::OutputRows;
  TmaWindow<C::Pipeline::Slots, C::SlotBytes, C::TransactionBytes> window(
      shared, barriers, input.maps + (C::TileLocalMaps ? blockIdx.x : 0));
  auto issue = [&](int32_t slot, int32_t in) { window.issue(slot, tileColumn, tileRow, schedule.input_plane(in)); };
  pipeline.prefetch(issue);

  // B fragments: kept in registers, staged in shared memory, or read from L1.
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % WARP_SIZE;
  const uint32_t *globalWeights = reinterpret_cast<const uint32_t *>(weight.wData) + lane;
  constexpr int32_t LaneWords = C::Weights.bytes() / (4 * WARP_SIZE);
  [[maybe_unused]] uint32_t registerWeights[C::Source == WeightSource::Registers ? LaneWords : 1];
  [[maybe_unused]] uint32_t sharedWeights = 0U;
  if constexpr (C::Source == WeightSource::Registers) {
#pragma unroll
    for (int32_t word = 0; word < LaneWords; ++word) {
      registerWeights[word] = __ldg(globalWeights + word * WARP_SIZE);
    }
  } else if constexpr (C::Source == WeightSource::Shared) {
    auto *destination = reinterpret_cast<uint4 *>(shared + C::WeightOffset);
    const auto *source = reinterpret_cast<const uint4 *>(weight.wData);
    for (int32_t i = threadIdx.x; i < C::Weights.bytes() / 16; i += C::Threads) {
      destination[i] = source[i];
    }
    sharedWeights = static_cast<uint32_t>(__cvta_generic_to_shared(shared + C::WeightOffset)) + 4 * lane;
  }
  auto loadB = [&](int32_t index) -> uint32_t {
    if constexpr (C::Source == WeightSource::Registers) {
      return registerWeights[index];
    } else if constexpr (C::Source == WeightSource::Shared) {
      return load_shared_u32(sharedWeights + 4 * WARP_SIZE * index);
    } else {
      return __ldg(globalWeights + WARP_SIZE * index);
    }
  };

  Mma mma;
  auto thread = mma.get_slice(lane);
  // A warp's output tile: (group, column in group, row).
  constexpr auto warpTile = make_shape(_16{}, Int<P>{}, Int<C::WarpRows>{});
  auto coordinates = thread.partition_C(make_identity_tensor(warpTile));
  auto accumulatorShape = append(replace<3>(shape(coordinates), make_shape(Int<C::RowsPerBatch>{}, Int<C::Batches>{})),
                                 Int<C::L>{});
  AccumulatorFile<C::L, decltype(make_tensor<int32_t>(accumulatorShape))> accumulators{
      make_tensor<int32_t>(accumulatorShape)};
  accumulators.clear();
  auto fragmentB = thread.partition_fragment_B(make_tensor(make_smem_ptr(shared), make_shape(_8{}, _32{})));
  auto packedB = recast<uint32_t>(fragmentB);

  const int32_t warp = static_cast<int32_t>(threadIdx.x) / WARP_SIZE;
  auto accumulate = [&]<int32_t FirstWeight, int32_t FirstAccumulator, int32_t Count>(int32_t slot)
                        __attribute__((always_inline)) {
#pragma unroll
    for (int32_t k = 0; k < C::KBlocks; ++k) {
#pragma unroll
      for (int32_t tess = 0; tess < 2; ++tess) {
        core::EmulationType *base = window.data(slot) + (warp * C::WarpRows + k * C::RowsPerK) * SC + tess * P;
        auto sharedA = make_tensor(
            make_smem_ptr(base),
            make_layout(make_shape(_16{}, make_shape(Int<P>{}, Int<C::RowsPerK>{}), Int<C::Batches>{},
                                   Int<C::RowsPerBatch>{}),
                        make_stride(Int<P>{}, make_stride(_1{}, Int<SC>{}), Int<C::RowsPerBatch * SC>{}, Int<SC>{})));
        auto partitionA = thread.partition_A(sharedA);
#pragma unroll
        for (int32_t batch = 0; batch < C::Batches; ++batch) {
          auto batchA = partitionA(_, _, _, batch, _);
          auto fragmentA = thread.make_fragment_A(batchA);
          copy(recast<uint32_t>(batchA), recast<uint32_t>(fragmentA));
          static_for<Count>([&](auto c) __attribute__((always_inline)) {
            constexpr int32_t Slice = M - 1 - (FirstWeight + decltype(c)::value);
            accumulators.template visit<FirstAccumulator + decltype(c)::value>(
                [&](auto accumulator) __attribute__((always_inline)) {
#pragma unroll
                  for (int32_t n8 = 0; n8 < C::N8; ++n8) {
                    const int32_t sourceN8 = C::N8 * Slice + n8;
                    packedB(0) = loadB(layout.word_index(tess, k, sourceN8, 0, 0) / WARP_SIZE);
                    packedB(1) = loadB(layout.word_index(tess, k, sourceN8, 1, 0) / WARP_SIZE);
#pragma unroll
                    for (int32_t row = 0; row < C::RowsPerBatch; ++row) {
                      auto rowAccumulator = accumulator(_, _, n8, make_coord(row, batch));
                      gemm(mma, fragmentA(_, _, _, row), fragmentB,
                           make_tensor(rowAccumulator.data(), append<3>(rowAccumulator.layout())));
                    }
                  }
                });
          });
        }
      }
    }
  };

  const int32_t outputRowStart = static_cast<int32_t>(blockIdx.y) * C::OutputRows + warp * C::WarpRows + C::Halo;
  const int32_t outputColumnStart = static_cast<int32_t>(blockIdx.x) * C::OutputCols + C::Halo;
  const int32_t validRows = min(C::WarpRows, output.ny - C::Halo - outputRowStart);
  const int32_t validColumns = min(C::OutputCols, output.nx - C::Halo - outputColumnStart);
  const bool fullTile = validRows == C::WarpRows && validColumns == C::OutputCols;
  const int32_t pointerRow = validRows > 0 ? outputRowStart : C::Halo; // keep the base pointer in bounds
  auto inBounds = lazy::transform(coordinates, [=](auto coordinate) {
    return get<0>(coordinate) * P + get<1>(coordinate) < validColumns && get<2>(coordinate) < validRows;
  });

  uint32_t planeMask = 0U; // non-zero output slices
  int32_t mssMin = 0;      // extrema of the most significant slice (zero-inclusive)
  int32_t mssMax = 0;
  auto emit = [&](int32_t diagonal) __attribute__((always_inline)) {
    const int32_t plane = schedule.output_plane(diagonal);
    if (plane < 0 || plane > schedule.dataSlices) {
      accumulators.propagate();
      return;
    }
    auto threadOutput = thread.partition_C(
        make_tensor(make_gmem_ptr(output.ptr(plane, outputColumnStart, pointerRow)),
                    make_layout(warpTile, make_stride(Int<P>{}, _1{}, output.rowStrideBytes))));
    auto digits = make_fragment_like<core::EmulationType>(accumulators.prototype());
    accumulators.template split<true>(digits);
    if (fullTile) {
      copy(digits, threadOutput);
    } else {
      copy_if(inBounds, digits, threadOutput);
    }
    auto stored = make_zip_tensor(digits, inBounds);
    if (plane == schedule.dataSlices) {
      const auto extrema = reduce(stored, make_tuple(0, 0), [](auto current, auto digitAndValid) {
        if (get<1>(digitAndValid)) {
          const int32_t digit = get<0>(digitAndValid);
          get<0>(current) = min(get<0>(current), digit);
          get<1>(current) = max(get<1>(current), digit);
        }
        return current;
      });
      mssMin = get<0>(extrema);
      mssMax = get<1>(extrema);
      planeMask |= (mssMin | mssMax) != 0 ? 1U << plane : 0U;
    } else {
      const uint32_t bits = reduce(stored, 0U, [](uint32_t current, auto digitAndValid) {
        return get<1>(digitAndValid) ? current | static_cast<uint32_t>(get<0>(digitAndValid)) : current;
      });
      planeMask |= bits != 0U ? 1U << plane : 0U;
    }
  };

  pipeline.run(window, issue, accumulate, emit);
  publish_metadata(output.info, planeMask, mssMin, mssMax);
}

// Calls function.template operator()<Side, M>() for a supported (side, slices).
template <class Function> auto with_2d_kernel(int32_t side, int32_t slices, Function &&function) {
  constexpr std::pair<int32_t, int32_t> kernels[] = {{3, 1},  {3, 2},  {3, 9},  {7, 1},  {7, 2},
                                                     {7, 9},  {15, 1}, {15, 2}, {15, 3}, {15, 9}};
  using Result = decltype(function.template operator()<3, 1>());
  Result result{};
  bool found = false;
  static_for<std::size(kernels)>([&](auto i) {
    constexpr auto kernel = kernels[decltype(i)::value];
    if (!found && side == kernel.first && slices == kernel.second) {
      result = function.template operator()<kernel.first, kernel.second>();
      found = true;
    }
  });
  return std::make_pair(found, result);
}

} // namespace kernels

cudaError_t launch_stencil_2d(PackedStencil &stencil, SlicedTensor &output, SlicedTensor &input, cudaStream_t stream) {
  const auto [found, status] = kernels::with_2d_kernel(stencil.side, stencil.slices, [&]<int32_t Side, int32_t M>() {
    using C = kernels::Config2d<Side, M>;
    const auto kernel = kernels::stencil_2d<Side, M>;
    cudaError_t error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SharedBytes);
    if (error != cudaSuccess) {
      return error;
    }
    const PackedStencilArg weight = stencil.arg();
    kernels::preamble<M><<<1, 1, 0, stream>>>(output.descriptor(), weight, input.descriptor());
    const dim3 grid(cdiv(output.nx() - 2 * C::Halo, C::OutputCols), cdiv(output.ny() - 2 * C::Halo, C::OutputRows));
    kernel<<<grid, C::Threads, C::SharedBytes, stream>>>(output.view(), weight, input.tma_view());
    return cudaGetLastError();
  });
  return found ? status : cudaErrorInvalidValue;
}

CUresult encode_input_tensor_maps_2d(const PackedStencil &stencil, SlicedTensor &tensor,
                                     std::vector<CUtensorMap> &maps) {
  const auto [found, status] = kernels::with_2d_kernel(stencil.side, stencil.slices, [&]<int32_t Side, int32_t M>() {
    using C = kernels::Config2d<Side, M>;
    const int32_t nx = tensor.nx();
    const int32_t tiles = C::TileLocalMaps ? cdiv(nx - 2 * C::Halo, C::OutputCols) : 1;
    const uint64_t strides[2] = {tensor.row_pitch_bytes(), tensor.slice_stride_bytes()};
    const uint32_t box[3] = {C::SharedColumns / C::TmaElementBytes, C::OutputRows + 2 * C::Halo, 1};
    const uint32_t elementStrides[3] = {1, 1, 1};
    maps.assign(static_cast<size_t>(tiles), CUtensorMap{});
    for (int32_t tile = 0; tile < tiles; ++tile) {
      const int32_t left = tile * C::OutputCols;
      const int32_t columns = C::TileLocalMaps ? std::min(nx - left, C::OutputCols + 2 * C::Halo) : nx;
      const uint64_t extents[3] = {static_cast<uint64_t>(cdiv(columns, C::TmaElementBytes)),
                                   static_cast<uint64_t>(tensor.ny()),
                                   static_cast<uint64_t>(tensor.numerical_slices())};
      const CUresult result = cuTensorMapEncodeTiled(
          &maps[static_cast<size_t>(tile)],
          C::TmaElementBytes == 2 ? CU_TENSOR_MAP_DATA_TYPE_UINT16 : CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
          tensor.data() + left, extents, strides, box, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (result != CUDA_SUCCESS) {
        return result;
      }
    }
    return CUDA_SUCCESS;
  });
  return found ? status : CUDA_ERROR_INVALID_VALUE;
}

} // namespace ozaki::gpu
