#ifndef HEADER_OZI_CUH
#define HEADER_OZI_CUH

#include "../core/ozconcepts.hpp"
#include "gpu_checks.cuh"

#include <concepts>

#ifndef RDC_THREADS_PER_BLOCK
#define RDC_THREADS_PER_BLOCK 512
#endif

namespace ozigpu {

using namespace ozi;

#ifdef __CUDACC__
// for vectorized load store ops
template <typename T> struct FourValues {
  T a[4];
};

template <std::integral T> __host__ __device__ __forceinline__ int32_t ozaki_scaling_factor(int32_t *v) {
  int32_t factor = (CHAR_BIT * sizeof(T) - 1) - *v;
  *v -= CHAR_BIT * sizeof(T) - 1;
  return factor;
}

template <std::floating_point T>
  requires(sizeof(T) > 1)
struct OzakiFP_Int8 {
  using data_type = T;
  using target_type = int8_t;
  using acc_type = int32_t;
};

__device__ __forceinline__ int8_t down_cast(double a, int32_t scale) {
  double a_scaled = scalbn(a, scale);
  int32_t temp;
  asm("cvt.rni.sat.s8.f64 %0, %1;" : "=r"(temp) : "d"(a_scaled));
  return static_cast<int8_t>(max(-127, temp));
}
__device__ __forceinline__ int8_t down_cast(float a, int32_t scale) {
  float a_scaled = scalbn(a, scale);
  int32_t temp;
  asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(temp) : "f"(a_scaled));
  return static_cast<int8_t>(max(-127, temp));
}

template <std::floating_point T, std::integral U> __device__ __forceinline__ T up_cast(U a, int32_t scale) {
  return scalbn(static_cast<T>(a), -scale);
}

template <typename T> __global__ void fill(T *__restrict__ in, const T val, size_t n) {
  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n)
    in[tid] = val;
}

template <OzakiData T>
__global__ void abs_max_nextpow2_reduce(int32_t *__restrict__ out, const T *__restrict__ in, size_t n) {
  constexpr uint32_t WARPS_PER_BLOCK = RDC_THREADS_PER_BLOCK / WARP_SIZE;
  constexpr uint32_t BLOCK_STRIDE = RDC_THREADS_PER_BLOCK;
  __shared__ T smem[WARPS_PER_BLOCK];

  const uint32_t GRID_SIZE = BLOCK_STRIDE * gridDim.x;
  uint32_t i = threadIdx.x + blockIdx.x * BLOCK_STRIDE;

  T local_val = static_cast<T>(0);
#pragma unroll 4
  while (4 * i < n) { // I assume that n is a multiple of four.
    const T *src = in + 4 * i;
    FourValues<T> vals = *reinterpret_cast<const FourValues<T> *>(src);
    local_val = max(max(max(abs(vals.a[0]), abs(vals.a[1])), abs(vals.a[2])), abs(vals.a[3]));
    i += GRID_SIZE;
  }

  constexpr uint32_t FULL_MASK = 0xFFFF'FFFF;
  local_val = max(local_val, __shfl_down_sync(FULL_MASK, local_val, 16));
  local_val = max(local_val, __shfl_down_sync(FULL_MASK, local_val, 8));
  local_val = max(local_val, __shfl_down_sync(FULL_MASK, local_val, 4));
  local_val = max(local_val, __shfl_down_sync(FULL_MASK, local_val, 2));
  local_val = max(local_val, __shfl_down_sync(FULL_MASK, local_val, 1));

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
    for (uint32_t offset = WARPS_PER_BLOCK / 2; offset > 0; offset /= 2)
      local_val = max(local_val, __shfl_down_sync(mask, local_val, offset));
    if (LANE_ID == 0) {
      int32_t block_partial_result = static_cast<int32_t>(ceil(log2(local_val)));
      atomicMax(out, block_partial_result);
    }
  }
}

template <OzakiPair O>
__global__ void slice(typename O::target_type *__restrict__ out, const typename O::data_type *__restrict__ in,
                      const int32_t *__restrict__ power, const int32_t num_slices, const size_t n) {
  using Data = typename O::data_type;
  using Target = typename O::target_type;
  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n / 4)
    return;

  int32_t local_power = *power;

  FourValues<Data> ins = reinterpret_cast<const FourValues<Data> *>(in)[tid];
  FourValues<Target> outs{};

#pragma unroll 4
  for (int32_t s = 0; s < num_slices; ++s) {
    int32_t shift_amount = ozaki_scaling_factor<Target>(&local_power);
    const size_t flat_offset = static_cast<size_t>(s) * (n / 4) + static_cast<size_t>(tid);
#pragma unroll
    for (int32_t v = 0; v < 4; ++v) {
      const Target v_converted = down_cast(ins.a[v], shift_amount);
      ins.a[v] -= up_cast<Data, Target>(v_converted, shift_amount);
      outs.a[v] = v_converted;
    }
    reinterpret_cast<FourValues<Target> *>(out)[flat_offset] = outs;
  }
}

template <OzakiPair O>
__global__ void mend(typename O::data_type *__restrict__ out, const typename O::target_type *__restrict__ in,
                     const int32_t *__restrict__ power, const int32_t num_slices, const size_t n) {
  using Data = typename O::data_type;
  using Target = typename O::target_type;

  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n / 4)
    return;

  int32_t local_power = *power;

  FourValues<Data> outs{};
  FourValues<Target> ins{};

#pragma unroll 4
  for (int32_t s = 0; s < num_slices; ++s) {
    int32_t shift_amount = ozaki_scaling_factor<Target>(&local_power);
    const size_t flat_offset = static_cast<size_t>(s) * (n / 4) + static_cast<size_t>(tid);
    ins = reinterpret_cast<const FourValues<Target> *>(in)[flat_offset];
#pragma unroll
    for (int32_t v = 0; v < 4; ++v) {
      outs.a[v] += up_cast<Data, Target>(ins.a[v], shift_amount);
    }
  }

  reinterpret_cast<FourValues<Data> *>(out)[tid] = outs;
}

CUDA_GLOBAL void ozi_mul_stencil(int8_t *__restrict__ out, const int8_t *__restrict__ stencil,
                                 const int8_t *__restrict__ data, const int32_t num_slices, const size_t m,
                                 const size_t n, const size_t halo_size);

#endif
} // namespace ozigpu

void gpu_ozaki_decomposition(int8_t *slices, int32_t *exp, const double *input, const int num_slices, size_t size_flat);

#endif // HEADER_OZI_CUH
