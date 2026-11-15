#include "stencil_common.cuh"
#include "stencil_kernels.cuh"

#include "weight_layout.hpp"

#include <cute/algorithm/gemm.hpp>

namespace ozaki::gpu {
namespace kernels {

constexpr int32_t Footprint = 5;
constexpr int32_t FootprintHalo = Footprint / 2;
constexpr int32_t Threads3d = 256;
constexpr int32_t OutputTileX = 192;
constexpr int32_t OutputTileY = 4;
constexpr int32_t SharedColumns = 224;
constexpr int32_t SharedRows = OutputTileY + 2 * FootprintHalo;
constexpr int32_t SharedPlaneStride = SharedColumns * SharedRows;

template <int32_t M> struct Config3d {
  static constexpr int32_t OutputTileZ = M <= 2 ? 8 : 2;
  static constexpr int32_t SharedPlanes = OutputTileZ + 2 * FootprintHalo;
  static constexpr int32_t SlotBytes = SharedPlaneStride * SharedPlanes;
  // Ping-pong for M <= 2; for M = 9 seven diagonals live in registers and each
  // input stays resident for two diagonals.
  static constexpr int32_t L = M > 2 ? M - 2 : 1;
  using Pipeline = DiagonalSchedule<M, L>;
  static constexpr int32_t InputBytes = Pipeline::Slots * SlotBytes;
  static constexpr Mma3dWeightLayout Weights{M};
  static constexpr int32_t SharedBytes = InputBytes + Weights.bytes();
  static_assert(Pipeline::Slots == 2);
};

CUDA_DEVICE CUDA_INLINE uint16_t load_shared_u16(uint32_t address) {
  uint16_t value;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(value) : "r"(address));
  return value;
}

template <int32_t Parity, int32_t First>
CUDA_DEVICE CUDA_INLINE uint32_t load_four(uint32_t slot, int32_t z, int32_t warpY, int32_t mmaRow, int32_t y) {
  constexpr int32_t Offset = 6 * Parity + First;
  const uint32_t address = slot + z * SharedPlaneStride + (warpY + y) * SharedColumns + 12 * mmaRow + Offset;
  if constexpr (Offset % 4 == 0) {
    return load_shared_u32(address);
  } else {
    return static_cast<uint32_t>(load_shared_u16(address)) | static_cast<uint32_t>(load_shared_u16(address + 2)) << 16;
  }
}

template <int32_t Parity>
CUDA_DEVICE CUDA_INLINE uint32_t load_tail(uint32_t slot, int32_t z, int32_t warpY, int32_t mmaRow, int32_t y) {
  return load_shared_u16(slot + z * SharedPlaneStride + (warpY + y) * SharedColumns + 12 * mmaRow + 6 * Parity + 4);
}

template <int32_t Parity, int32_t Tess, class Fragment>
CUDA_DEVICE CUDA_INLINE void load_a(Fragment &a, uint32_t slot, int32_t z, int32_t warpY, int32_t lane) {
  const int32_t mmaRow = lane >> 2;
  const int32_t kGroup = lane & 3;
  auto tail = [&](int32_t row) __attribute__((always_inline)) -> uint32_t {
    switch (kGroup) {
    case 0:
      return load_four<Parity, 0>(slot, z, warpY, row, 4);
    case 1:
      return load_tail<Parity>(slot, z, warpY, row, 0) | load_tail<Parity>(slot, z, warpY, row, 1) << 16;
    case 2:
      return load_tail<Parity>(slot, z, warpY, row, 2) | load_tail<Parity>(slot, z, warpY, row, 3) << 16;
    default:
      return load_tail<Parity>(slot, z, warpY, row, 4);
    }
  };
  if constexpr (Tess == 0) {
    a(0) = load_four<Parity, 0>(slot, z, warpY, mmaRow, kGroup);
    a(1) = load_four<Parity, 0>(slot, z, warpY, mmaRow + 8, kGroup);
    a(2) = tail(mmaRow);
    a(3) = tail(mmaRow + 8);
  } else {
    a(0) = load_four<Parity, 6>(slot, z, warpY, mmaRow, kGroup);
    a(1) = load_four<Parity, 6>(slot, z, warpY, mmaRow + 8, kGroup);
    a(2) = kGroup == 0 ? load_four<Parity, 6>(slot, z, warpY, mmaRow, 4) : 0U;
    a(3) = kGroup == 0 ? load_four<Parity, 6>(slot, z, warpY, mmaRow + 8, 4) : 0U;
  }
}

template <class A, class B, class Accumulator>
CUDA_DEVICE CUDA_INLINE void mma_s8(const A &a, const B &b, Accumulator &&c) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c(0)), "+r"(c(1)), "+r"(c(2)), "+r"(c(3))
      : "r"(a(0)), "r"(a(1)), "r"(a(2)), "r"(a(3)), "r"(b(0)), "r"(b(1)));
}

template <int32_t Side, int32_t M>
__global__ __launch_bounds__(Threads3d, 2) void stencil_3d(SlicedTensorView output, PackedStencilArg weight,
                                                           const __grid_constant__ SlicedTensorTmaView input) {
  constexpr int32_t halo = Side / 2;
#pragma nv_diag_suppress static_var_with_dynamic_init
  using C = Config3d<M>;
  using namespace cute;
  constexpr int32_t TileZ = C::OutputTileZ;
  constexpr Mma3dWeightLayout layout = C::Weights;

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

  const int32_t tileX = static_cast<int32_t>(blockIdx.x) * OutputTileX;
  const int32_t tileY = static_cast<int32_t>(blockIdx.y) * OutputTileY;
  const int32_t tileZ = static_cast<int32_t>(blockIdx.z) * TileZ;
  TmaWindow<C::Pipeline::Slots, C::SlotBytes, C::SlotBytes> window(shared, barriers, input.maps);
  auto issue = [&](int32_t slot, int32_t in) {
    window.issue(slot, tileX, tileY, tileZ, schedule.input_plane(in));
  };
  pipeline.prefetch(issue);

  auto *sharedWeights = reinterpret_cast<uint4 *>(shared + C::InputBytes);
  const auto *globalWeights = reinterpret_cast<const uint4 *>(weight.wData);
  for (int32_t i = threadIdx.x; i < C::Weights.bytes() / 16; i += Threads3d) {
    sharedWeights[i] = globalWeights[i];
  }
  __syncthreads();

  const int32_t lane = static_cast<int32_t>(threadIdx.x) % WARP_SIZE;
  const int32_t warp = static_cast<int32_t>(threadIdx.x) / WARP_SIZE;
  Mma mma;
  auto thread = mma.get_slice(lane);
  auto coordinates = thread.partition_C(make_identity_tensor(make_shape(_16{}, _8{}, Int<TileZ>{})));
  auto accumulatorShape = append(shape(coordinates), Int<C::L>{});
  AccumulatorFile<C::L, decltype(make_tensor<int32_t>(accumulatorShape))> accumulators{
      make_tensor<int32_t>(accumulatorShape)};
  accumulators.clear();
  auto fragmentA = thread.partition_fragment_A(make_tensor(make_smem_ptr(shared), make_shape(_16{}, _32{})));
  auto fragmentB = thread.partition_fragment_B(make_tensor(make_smem_ptr(shared), make_shape(_8{}, _32{})));
  auto packedA = recast<uint32_t>(fragmentA);
  auto packedB = recast<uint32_t>(fragmentB);

  auto accumulate = [&]<int32_t FirstWeight, int32_t FirstAccumulator, int32_t Count>(int32_t slot)
                        __attribute__((always_inline)) {
    const uint32_t slotAddress = static_cast<uint32_t>(__cvta_generic_to_shared(window.data(slot)));
    const uint32_t weightAddress = slotAddress + C::InputBytes - slot * C::SlotBytes + 4 * lane;
    static_for<C::SharedPlanes>([&](auto inputZ) __attribute__((always_inline)) {
      static_for<2>([&](auto tess) __attribute__((always_inline)) {
        constexpr int32_t Tess = decltype(tess)::value;
        if ((warp & 1) == 0) {
          load_a<0, Tess>(packedA, slotAddress, inputZ, warp >> 1, lane);
        } else {
          load_a<1, Tess>(packedA, slotAddress, inputZ, warp >> 1, lane);
        }
        static_for<Count>([&](auto c) __attribute__((always_inline)) {
          constexpr int32_t Slice = M - 1 - (FirstWeight + decltype(c)::value);
          accumulators.template visit<FirstAccumulator + decltype(c)::value>(
              [&](auto &&accumulator) __attribute__((always_inline)) {
                static_for<TileZ>([&](auto outputZ) __attribute__((always_inline)) {
                  constexpr int32_t CoefficientZ = decltype(inputZ)::value - decltype(outputZ)::value;
                  if constexpr (CoefficientZ >= 0 && CoefficientZ < Footprint) {
                    constexpr int32_t Word0 = layout.word_index(CoefficientZ, Tess, Slice, 0, 0);
                    constexpr int32_t Word1 = layout.word_index(CoefficientZ, Tess, Slice, 1, 0);
                    packedB(0) = load_shared_u32(weightAddress + 4 * Word0);
                    packedB(1) = load_shared_u32(weightAddress + 4 * Word1);
                    auto planeAccumulator = accumulator(_, _, _, outputZ);
                    if constexpr (M == 9) {
                      mma_s8(packedA, packedB, planeAccumulator);
                    } else {
                      gemm(mma, fragmentA, fragmentB, planeAccumulator);
                    }
                  }
                });
              });
        });
      });
    });
  };

  uint32_t planeMask = 0U;
  int32_t mssMin = 0;
  int32_t mssMax = 0;
  // Stores one output digit plane; TrackMss also records the metadata.
  auto store = [&]<bool TrackMss>(int32_t plane, const auto &digits) __attribute__((always_inline)) {
    const int32_t startX = static_cast<int32_t>(blockIdx.x) * OutputTileX + halo;
    const int32_t outputY = static_cast<int32_t>(blockIdx.y) * OutputTileY + (warp >> 1) + halo;
    const int32_t startZ = static_cast<int32_t>(blockIdx.z) * TileZ + halo;
    const int32_t validX = max(0, min(OutputTileX, output.nx - halo - startX));
    const int32_t validZ = max(0, min(TileZ, output.nz - halo - startZ));
    const int32_t laneRow = lane >> 2;
    const int32_t lanePair = lane & 3;
    uint32_t nonzero = 0U;
    if (lanePair < 3 && outputY < output.ny - halo) {
      core::EmulationType *base = output.ptr(plane, startX, outputY, startZ);
      static_for<TileZ>([&](auto z) __attribute__((always_inline)) {
        if (z < validZ) {
          auto planeDigits = digits(_, _, _, z);
          static_for<2>([&](auto half) __attribute__((always_inline)) {
            const int32_t localX = 12 * (laneRow + 8 * half) + 6 * (warp & 1) + 2 * lanePair;
            if (localX < validX) {
              const core::EmulationType first = planeDigits(2 * half);
              const core::EmulationType second = planeDigits(2 * half + 1);
              const bool both = localX + 1 < validX;
              core::EmulationType *destination = base + z * output.planeStrideBytes + localX;
              if (halo % 2 == 0 && both) { // two-byte aligned pair
                *reinterpret_cast<uint16_t *>(destination) =
                    static_cast<uint8_t>(first) | static_cast<uint16_t>(static_cast<uint8_t>(second)) << 8;
              } else {
                destination[0] = first;
                if (both) {
                  destination[1] = second;
                }
              }
              const int32_t secondValue = both ? second : 0;
              nonzero |= static_cast<uint8_t>(first) | static_cast<uint8_t>(secondValue);
              if constexpr (TrackMss) {
                mssMin = min(mssMin, min(static_cast<int32_t>(first), secondValue));
                mssMax = max(mssMax, max(static_cast<int32_t>(first), secondValue));
              }
            }
          });
        }
      });
    }
    planeMask |= nonzero != 0U ? 1U << plane : 0U;
  };

  auto emit = [&](int32_t diagonal) __attribute__((always_inline)) {
    const int32_t plane = schedule.output_plane(diagonal);
    if (plane < 0 || plane > schedule.dataSlices) {
      accumulators.propagate();
      return;
    }
    auto digits = make_tensor<core::EmulationType>(shape(accumulators.prototype()));
    accumulators.template split<true>(digits);
    if (plane == schedule.dataSlices) {
      store.template operator()<true>(plane, digits);
    } else {
      store.template operator()<false>(plane, digits);
    }
  };

  pipeline.run(window, issue, accumulate, emit);
  publish_metadata(output.info, planeMask, mssMin, mssMax);
}

} // namespace kernels

namespace {

template <int32_t Side, int32_t M>
cudaError_t launch_3d(PackedStencil &stencil, SlicedTensor &output, SlicedTensor &input, cudaStream_t stream) {
  using C = kernels::Config3d<M>;
  const auto kernel = kernels::stencil_3d<Side, M>;
  const cudaError_t error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SharedBytes);
  if (error != cudaSuccess) {
    return error;
  }
  const PackedStencilArg weight = stencil.arg();
  kernels::preamble<M><<<1, 1, 0, stream>>>(output.descriptor(), weight, input.descriptor());
  constexpr int32_t halo = Side / 2;
  const dim3 grid(cdiv(output.nx() - 2 * halo, kernels::OutputTileX), cdiv(output.ny() - 2 * halo, kernels::OutputTileY),
                  cdiv(output.nz() - 2 * halo, C::OutputTileZ));
  kernel<<<grid, kernels::Threads3d, C::SharedBytes, stream>>>(output.view(), weight, input.tma_view());
  return cudaGetLastError();
}

} // namespace

cudaError_t launch_stencil_3d(PackedStencil &stencil, SlicedTensor &output, SlicedTensor &input, cudaStream_t stream) {
  const auto launch = [&]<int32_t Side>() {
    switch (stencil.slices) {
    case 1:
      return launch_3d<Side, 1>(stencil, output, input, stream);
    case 2:
      return launch_3d<Side, 2>(stencil, output, input, stream);
    case 9:
      return launch_3d<Side, 9>(stencil, output, input, stream);
    default:
      return cudaErrorInvalidValue;
    }
  };
  switch (stencil.side) {
  case 3:
    return launch.template operator()<3>();
  case 5:
    return launch.template operator()<5>();
  default:
    return cudaErrorInvalidValue;
  }
}

CUresult encode_input_tensor_maps_3d(const PackedStencil &stencil, SlicedTensor &tensor,
                                     std::vector<CUtensorMap> &maps) {
  const int32_t tileZ = stencil.slices <= 2 ? 8 : 2;
  const uint64_t extents[4] = {static_cast<uint64_t>(tensor.nx()), static_cast<uint64_t>(tensor.ny()),
                               static_cast<uint64_t>(tensor.nz()), static_cast<uint64_t>(tensor.numerical_slices())};
  const uint64_t strides[3] = {tensor.row_pitch_bytes(), tensor.plane_stride_bytes(), tensor.slice_stride_bytes()};
  const uint32_t box[4] = {kernels::SharedColumns, kernels::SharedRows,
                           static_cast<uint32_t>(tileZ + 2 * kernels::FootprintHalo), 1};
  const uint32_t elementStrides[4] = {1, 1, 1, 1};
  maps.assign(1, CUtensorMap{});
  return cuTensorMapEncodeTiled(&maps[0], CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, tensor.data(), extents, strides, box,
                                elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

} // namespace ozaki::gpu
