#ifndef HEADER_OZI_KERNELS_CUH
#define HEADER_OZI_KERNELS_CUH

#include "../core/oztraits.hpp"
#include "gpu_checks.cuh"
#include "ozi.cuh"

#include <climits>
#include <concepts>
#include <cuda_runtime.h>
#include <limits>
#include <type_traits>

#ifdef __GNUC__
#include <quadmath.h>
#endif

#ifndef RDC_THREADS_PER_BLOCK
#define RDC_THREADS_PER_BLOCK 512
#endif

namespace ozigpu {
using namespace ozi;
template <typename T> __global__ void fill(T *__restrict__ in, const T val, size_t n) {
  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n)
    in[tid] = val;
}

template <typename T, typename ScaleT = double> __global__ void scale(T *__restrict__ in, const ScaleT val, size_t n) {
  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n)
    in[tid] = static_cast<T>(static_cast<ScaleT>(in[tid]) * val);
}

// Applies a scalar correction to boundary points only.
// A point is a boundary point when its thread index falls outside the
// interior region defined by the stencil offsets and shape — matching the
// same check_dim predicate used in stencil_v1.
// Interior points (in_bounds == true) are left unchanged.
template <typename T, typename ScaleT, size_t D>
  requires(D >= 1 && D <= 3)
__global__ void correct_boundary(T *__restrict__ data, ScaleT correction,
                                 std::array<size_t, D> input_shape,
                                 std::array<size_t, D> stencil_offsets,
                                 std::array<size_t, D> stencil_shape) {
  int32_t i = static_cast<int32_t>(threadIdx.x + blockDim.x * blockIdx.x);
  int32_t j = 0, k = 0;
  if constexpr (D > 1)
    j = static_cast<int32_t>(threadIdx.y + blockDim.y * blockIdx.y);
  if constexpr (D > 2)
    k = static_cast<int32_t>(threadIdx.z + blockDim.z * blockIdx.z);

  bool valid_thread;
  if constexpr (D == 1)
    valid_thread = (i < static_cast<int32_t>(input_shape[0]));
  else if constexpr (D == 2)
    valid_thread = (j < static_cast<int32_t>(input_shape[0])) &&
                   (i < static_cast<int32_t>(input_shape[1]));
  else
    valid_thread = (k < static_cast<int32_t>(input_shape[0])) &&
                   (j < static_cast<int32_t>(input_shape[1])) &&
                   (i < static_cast<int32_t>(input_shape[2]));

  if (!valid_thread)
    return;

  // Mirror the check_dim predicate from stencil_v1: a thread is in_bounds iff
  // its index is in [offset, input_size - (stencil_size - offset)] for each dim.
  auto check_dim = [&](int32_t idx, size_t d) -> bool {
    return idx >= static_cast<int32_t>(stencil_offsets[d]) &&
           idx <= static_cast<int32_t>(input_shape[d] -
                                       (stencil_shape[d] - stencil_offsets[d]));
  };

  bool in_bounds;
  if constexpr (D == 1)
    in_bounds = check_dim(i, 0);
  else if constexpr (D == 2)
    in_bounds = check_dim(j, 0) && check_dim(i, 1);
  else
    in_bounds = check_dim(k, 0) && check_dim(j, 1) && check_dim(i, 2);

  if (in_bounds)
    return; // interior point: no correction needed

  size_t out_offset;
  if constexpr (D == 1)
    out_offset = static_cast<size_t>(i);
  else if constexpr (D == 2)
    out_offset = static_cast<size_t>(i) +
                 static_cast<size_t>(j) * input_shape[1];
  else
    out_offset = static_cast<size_t>(i) +
                 static_cast<size_t>(j) * input_shape[2] +
                 static_cast<size_t>(k) * input_shape[1] * input_shape[2];

  data[out_offset] = static_cast<T>(static_cast<ScaleT>(data[out_offset]) * correction);
}

template <typename T, typename Dst = T, typename Acc = T>
  requires(!std::is_same_v<Dst, __float128>) // atomic operations not supported for FP128
__global__ void mse(Dst *__restrict__ out, const T *__restrict__ x, const T *__restrict__ y, size_t n) {
  constexpr uint32_t ELEMENTS_PER_VECTORIZED_LOAD = 16 / sizeof(T);
  constexpr uint32_t WARPS_PER_BLOCK = RDC_THREADS_PER_BLOCK / WARP_SIZE;
  constexpr uint32_t BLOCK_STRIDE = RDC_THREADS_PER_BLOCK;
  using VectorizedT = struct {
    T a[ELEMENTS_PER_VECTORIZED_LOAD];
  };

  __shared__ T smem[WARPS_PER_BLOCK];

  const uint32_t GRID_SIZE = BLOCK_STRIDE * gridDim.x;
  size_t i = threadIdx.x + blockIdx.x * BLOCK_STRIDE;
  Acc local_val = static_cast<T>(0);

#pragma unroll 4
  while (ELEMENTS_PER_VECTORIZED_LOAD * i < n) {
    const VectorizedT xv = *reinterpret_cast<const VectorizedT *>(x + ELEMENTS_PER_VECTORIZED_LOAD * i);
    const VectorizedT yv = *reinterpret_cast<const VectorizedT *>(y + ELEMENTS_PER_VECTORIZED_LOAD * i);
#pragma unroll
    for (uint32_t j = 0; j < ELEMENTS_PER_VECTORIZED_LOAD; ++j) {
      local_val += static_cast<Acc>(xv[j] - yv[j]) * static_cast<Acc>(xv[j] - yv[j]);
    }
    i += GRID_SIZE;
  }

  constexpr uint32_t FULL_MASK = 0xFFFF'FFFF;
#pragma unroll
  for (uint32_t offset = 16; offset > 0; offset /= 2) {
    local_val += shfl_down_sync_wrapper(FULL_MASK, local_val, offset);
  }

  const uint32_t WARP_ID = threadIdx.x / WARP_SIZE;
  const uint32_t LANE_ID = threadIdx.x % WARP_SIZE;
  if (LANE_ID == 0) {
    smem[WARP_ID] = local_val;
  }
  __syncthreads();

  if ((WARP_ID == 0) && (LANE_ID < WARPS_PER_BLOCK)) {
    uint32_t mask = __activemask();
    local_val = smem[LANE_ID];
#pragma unroll
    for (uint32_t offset = WARPS_PER_BLOCK / 2; offset > 0; offset /= 2) {
      local_val += shfl_down_sync_wrapper(mask, local_val, offset);
    }
    if (LANE_ID == 0) {
      atomicAdd(out, static_cast<Dst>(local_val));
    }
  }
}

/** Finds the dynamic range [pos(MSB), pos(LSB)] of a set of numbers */
template <OzakiPair O>
__global__ void dyn_range(OzakiDescriptor<O> *__restrict__ out, const typename O::data_type *__restrict__ in,
                          size_t n) {
  using Data = typename O::data_type;

  constexpr uint32_t WARPS_PER_BLOCK = RDC_THREADS_PER_BLOCK / WARP_SIZE;
  constexpr uint32_t BLOCK_STRIDE = RDC_THREADS_PER_BLOCK;
  constexpr uint32_t ELEMENTS_PER_VECTORIZED_LOAD = 16 / sizeof(Data);
  using VectorizedData = struct {
    Data a[ELEMENTS_PER_VECTORIZED_LOAD];
  };
  __shared__ Data smem_max[WARPS_PER_BLOCK];
  __shared__ int32_t smem_min[WARPS_PER_BLOCK];

  const uint32_t GRID_SIZE = BLOCK_STRIDE * gridDim.x;
  size_t i = threadIdx.x + blockIdx.x * BLOCK_STRIDE;

  Data local_max_val = static_cast<Data>(0);
  int32_t local_min_val = std::numeric_limits<int32_t>::max();

  // thread coarsening reduction
#pragma unroll 4
  while (ELEMENTS_PER_VECTORIZED_LOAD * i < n) { // this assumes that the input's total size is a multiple of 16 bytes.
    const Data *src = in + ELEMENTS_PER_VECTORIZED_LOAD * i;
    VectorizedData vals = *reinterpret_cast<const VectorizedData *>(src);
#pragma unroll
    for (uint32_t j = 0; j < ELEMENTS_PER_VECTORIZED_LOAD; ++j) {
      Data abs = OzakiTraits<Data>::abs(vals.a[j]);
      local_max_val = OzakiTraits<Data>::max(local_max_val, abs);
      local_min_val = dyn_min_lsb(vals.a[j], local_min_val);
    }
    i += GRID_SIZE;
  }

  // block level reduction
  constexpr uint32_t FULL_MASK = 0xFFFF'FFFF;
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset /= 2) {
    local_max_val = OzakiTraits<Data>::max(local_max_val, shfl_down_sync_wrapper(FULL_MASK, local_max_val, offset));
    local_min_val = min(local_min_val, __shfl_down_sync(FULL_MASK, local_min_val, offset));
  }

  const uint32_t WARP_ID = threadIdx.x / WARP_SIZE;
  const uint32_t LANE_ID = threadIdx.x % WARP_SIZE;

  if (LANE_ID == 0) {
    smem_max[WARP_ID] = local_max_val;
    smem_min[WARP_ID] = local_min_val;
  }
  __syncthreads();

  // warp level reduction
  if ((WARP_ID == 0) && (LANE_ID < WARPS_PER_BLOCK)) {
    uint32_t mask = __activemask();
    local_max_val = smem_max[LANE_ID];
    local_min_val = smem_min[LANE_ID];
#pragma unroll
    for (uint32_t offset = WARPS_PER_BLOCK / 2; offset > 0; offset /= 2) {
      local_max_val = OzakiTraits<Data>::max(local_max_val, shfl_down_sync_wrapper(mask, local_max_val, offset));
      local_min_val = min(local_min_val, __shfl_down_sync(mask, local_min_val, offset));
    }
    if (LANE_ID == 0) {
      int32_t block_partial_result = log2_ceil_abs(local_max_val);
      atomicMax(&out->e_max, block_partial_result);
      atomicMin(&out->e_min, local_min_val);
    }
  }
}

/** Computes the Ozaki decomposition of a set of numbers and stores it in *out */
template <OzakiPair O>
__global__ void slice(typename O::target_type *__restrict__ out, const typename O::data_type *__restrict__ in,
                      OzakiDescriptor<O> *__restrict__ desc, const size_t n) {
  using Data = typename O::data_type;
  using Target = typename O::target_type;

  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n / 4)
    return;

  int32_t emax = desc->e_max;
  int32_t total_slices = desc->n_slices_input;

  if (desc->n_slices_input == 0) {
    total_slices = 2 + (desc->e_max - desc->e_min) / (CHAR_BIT * sizeof(Target) - 1);
    if (tid == 0) {
      desc->n_slices_input = total_slices; // benign race
    }
  }

  FourValues<Data> ins = reinterpret_cast<const FourValues<Data> *>(in)[tid];
  FourValues<Target> outs{{0, 0, 0, 0}};

#pragma unroll 2
  for (int32_t s = 0; s < total_slices; ++s) {
    const size_t flat_offset = static_cast<size_t>(s) * (n / 4) + static_cast<size_t>(tid);
    if (s != 0) {
      int32_t shift_amount = OzakiTraits<Target>::scaling_factor(emax);
      emax = OzakiTraits<Target>::slice_step(emax);

#pragma unroll
      for (int32_t v = 0; v < 4; ++v) {
        const Target v_converted = OzakiPairTraits<O>::down_cast(ins.a[v], shift_amount);
        ins.a[v] -= OzakiPairTraits<O>::up_cast(v_converted, shift_amount);
        outs.a[v] = v_converted;
      }
    }
    reinterpret_cast<FourValues<Target> *>(out)[flat_offset] = outs;
  }
}

/** Reconstructs (mends) the slices of an Ozaki decomposition */
template <OzakiPair O, typename OutputType = typename O::data_type>
__global__ void mend(OutputType *__restrict__ out, const typename O::target_type *__restrict__ in,
                     OzakiDescriptor<O> *__restrict__ desc, const size_t n) {
  using Data = typename O::data_type;
  using Target = typename O::target_type;

  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n / 4)
    return;

  int32_t global_max_exp = desc->e_max + (CHAR_BIT * sizeof(Target) - 1);
  int32_t total_slices = desc->n_slices_input;
  int32_t carry_idx = desc->carry_slice_index;

  constexpr int32_t SLICE_BITS = CHAR_BIT * sizeof(Target) - 1;

  FourValues<OutputType> outs{{0, 0, 0, 0}};
  FourValues<Target> ins{};

  for (int32_t logic_s = total_slices - 1; logic_s >= 0; --logic_s) {
    int32_t current_slice_exp = global_max_exp - (logic_s * SLICE_BITS);
    int32_t shift_amount = SLICE_BITS - current_slice_exp;

    const int32_t physical_slice = (carry_idx + logic_s) % total_slices;
    const size_t flat_offset = physical_slice * (n / 4) + static_cast<size_t>(tid);
    ins = reinterpret_cast<const FourValues<Target> *>(in)[flat_offset];

#pragma unroll
    for (int32_t v = 0; v < 4; ++v) {
      outs.a[v] += static_cast<OutputType>(OzakiPairTraits<O>::up_cast(ins.a[v], shift_amount));
    }
  }
  reinterpret_cast<FourValues<OutputType> *>(out)[tid] = outs;
}

/** State update kernel for integration */
template <OzakiPair O>
__global__ void update_ozaki_desc(OzakiDescriptor<O> *__restrict__ desc,
                                  int32_t min_shifts = 0) {
  if (threadIdx.x == 0) {
    bool should_shift = (desc->shift_required != 0) ||
                        (desc->n_shifts < min_shifts);
    if (should_shift) {
      int32_t total = desc->n_slices_input;
      desc->carry_slice_index = (desc->carry_slice_index + total - 1) % total;
      desc->n_shifts += 1;
      desc->shift_required = 0;
    }
    desc->just_shifted = should_shift ? 1 : 0;
  }
}
} // namespace ozigpu

#endif // HEADER_OZI_KERNELS_CUH
