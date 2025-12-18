#ifndef HEADER_ISL_GPU_KERNELS_CUH
#define HEADER_ISL_GPU_KERNELS_CUH

#include "../core/ozi.hpp"
#include "gpu_checks.cuh"
#include <limits>
#include <type_traits>

#ifndef RDC_THREADS_PER_BLOCK
#define RDC_THREADS_PER_BLOCK 512
#endif

#ifdef __CUDACC__
template <typename T> struct alignas(16) FourValues {
  T a[4];
};

template <typename T>
struct BitwiseTypeTraits;

template <>
struct BitwiseTypeTraits<float> {
    using UintType = uint32_t;
    using IntType  = int32_t;
    
    CUDA_DEVICE CUDA_FORCEINLINE static IntType as_int(float x) {
        return __float_as_int(x);
    }
    CUDA_DEVICE CUDA_FORCEINLINE static float as_float(IntType x) {
        return __int_as_float(x);
    }
};

template <>
struct BitwiseTypeTraits<double> {
    using UintType = uint64_t;
    using IntType  = int64_t;

    CUDA_DEVICE CUDA_FORCEINLINE static IntType as_int(double x) {
        return __double_as_longlong(x);
    }
    CUDA_DEVICE CUDA_FORCEINLINE static double as_float(IntType x) {
        return __longlong_as_double(x);
    }
};

template <typename T> CUDA_GLOBAL void fill(T *__restrict__ in, const T val, size_t n) {
  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n)
    in[tid] = val;
}

template <typename T> CUDA_GLOBAL void abs_max_nextpow2_reduce(int32_t *__restrict__ out, const T *__restrict__ in, size_t n) {
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

template <typename T, typename U> CUDA_DEVICE CUDA_FORCEINLINE U gpu_round_and_clamp(T a) {
  constexpr int32_t LO = static_cast<int32_t>(std::numeric_limits<U>::min()) + 1;
  constexpr int32_t HI = static_cast<int32_t>(std::numeric_limits<U>::max());
  int32_t r;
  if constexpr (sizeof(T) == 8) r = __double2int_rn(a); 
  else r = __float2int_rn(a);
  return static_cast<U>(max(LO, min(HI, r)));
}

template <>
CUDA_DEVICE CUDA_FORCEINLINE int8_t gpu_round_and_clamp(double a) {
  int32_t ret;
  asm("cvt.rni.sat.s8.f64 %0, %1;" : "=r"(ret) : "d"(a));
  return static_cast<int8_t>(max(-127, ret));
}
template <>
CUDA_DEVICE CUDA_FORCEINLINE int8_t gpu_round_and_clamp(float a) {
  int32_t ret;
  asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(ret) : "f"(a));
  return static_cast<int8_t>(max(-127, ret));
}

template <typename T, typename U>
CUDA_GLOBAL void ozaki_slice(U *__restrict__ out, const T *__restrict__ in, const int32_t *__restrict__ power,
                             const int32_t num_slices, const size_t n) {
  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n / 4)
    return;

  const int32_t shift_step = static_cast<int32_t>(ozaki::effective_type_width<U>());
  int32_t shift_amount = shift_step - *power;

  FourValues<T> ins = reinterpret_cast<const FourValues<T> *>(in)[tid];

#pragma unroll 4
  for (int32_t s = 0; s < num_slices; ++s) {
    const size_t flat_offset = (static_cast<size_t>(s) * (n / 4) + static_cast<size_t>(tid)) * 4;
#pragma unroll
    for (int32_t v = 0; v < 4; ++v) {
      const T v_scaled = scalbn(ins.a[v], shift_amount);
      const U v_converted = gpu_round_and_clamp<T, U>(v_scaled);
      const T correction = scalbn(static_cast<T>(v_converted), -shift_amount);
      ins.a[v] -= correction;
      out[flat_offset + v] = v_converted;
    }
    shift_amount += shift_step;
  }
}

CUDA_GLOBAL void ozi_mul_stencil(
    int8_t *__restrict__ out,
    const int8_t *__restrict__ stencil,
    const int8_t *__restrict__ data,
    const int32_t num_slices,
    const size_t m, const size_t n, const size_t halo_size);

#endif

void gpu_ozaki_decomposition(int8_t *slices, int32_t *exp, const double *input, const int num_slices, size_t size_flat);

#endif // HEADER_ISL_GPU_KERNELS_CUH
