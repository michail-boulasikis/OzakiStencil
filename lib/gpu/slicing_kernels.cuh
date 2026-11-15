// Conversion between dense floating-point tensors and sliced int8 tensors.
//
// A value x is stored as int8 digits d_s with x ~= 128^rH * sum_s d_s 128^-(s+1),
// most significant digit in plane rR (rR = data slices); plane 0 holds the
// next digit (the error slice) and is not part of the value.
#ifndef OZAKI_GPU_SLICING_KERNELS_CUH
#define OZAKI_GPU_SLICING_KERNELS_CUH

#include <ozaki/gpu/digit_math.cuh>
#include <ozaki/gpu/sliced_tensor.cuh>

#include <ozaki/core/traits.hpp>

#include <limits>

namespace ozaki::gpu {

constexpr int32_t SLICING_THREADS = 256;

// Largest base-128 exponent over all values: the rH of the sliced tensor.
template <core::Real T>
__global__ __launch_bounds__(SLICING_THREADS) void _msb(int32_t *__restrict__ scale, const T *__restrict__ data,
                                                        size_t n) {
  // Zero values must not raise the exponent; start from the smallest one.
  int32_t local = std::numeric_limits<T>::min_exponent - std::numeric_limits<T>::digits + 1;
  for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    if (data[i] != static_cast<T>(0.0)) {
      int32_t exponent;
      core::RealTraits<T>::frexp(data[i], &exponent);
      local = max(local, exponent);
    }
  }
  local = __reduce_max_sync(0xFFFF'FFFFU, local);
  if (threadIdx.x % WARP_SIZE == 0) {
    atomicMax(scale, cdiv(local, core::SLICE_STEP));
  }
}

// Slices `src` into rR slices plus the error slice and fills the slice
// descriptor (non-zero slice mask, extrema of the most significant slice).
template <core::Real T>
__global__ __launch_bounds__(SLICING_THREADS) void _slice(SlicedTensorView dst, const T *__restrict__ src,
                                                          const int32_t *__restrict__ scale, int32_t rR, size_t n) {
  using Acc = T;
  const int32_t scaleStart = *scale;
  const int32_t exponentStart = scaleStart * core::SLICE_STEP;

  uint32_t mask = 0U;
  int32_t mssMin = std::numeric_limits<core::EmulationType>::max();
  int32_t mssMax = std::numeric_limits<core::EmulationType>::min();
  for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    // Relative to the scale, so the most significant digit cannot overflow.
    Acc residual = core::RealTraits<Acc>::scalbn(static_cast<Acc>(src[i]), -exponentStart);
    for (int32_t s = 0; s <= rR; s++) {
      const int32_t plane = rR - s;
      const int32_t castScale = core::SLICE_STEP * (s + 1);
      const core::EmulationType digit = core::RealTraits<Acc>::down_cast(residual, castScale);
      *dst.ptr_from_linear(plane, i) = digit;
      residual = residual - core::RealTraits<Acc>::up_cast(digit, castScale);
      mask |= digit != 0 ? 1U << plane : 0U;
      if (s == 0) {
        mssMax = max(mssMax, static_cast<int32_t>(digit));
        mssMin = min(mssMin, static_cast<int32_t>(digit));
      }
    }
  }

  mask = __reduce_or_sync(0xFFFF'FFFFU, mask);
  mssMin = __reduce_min_sync(0xFFFF'FFFFU, mssMin);
  mssMax = __reduce_max_sync(0xFFFF'FFFFU, mssMax);
  if (threadIdx.x % WARP_SIZE == 0) {
    atomicOr(&dst.info->sMask, mask);
    atomicMin(&dst.info->mssMin, mssMin);
    atomicMax(&dst.info->mssMax, mssMax);
  }
  if (threadIdx.x == 0) {
    dst.info->rH = scaleStart;
    dst.info->rL = scaleStart - rR + 1;
    dst.info->overflowSlices = 0;
  }
}

// Reconstructs the first `rR` data planes of `src` into `dst`.
template <core::Real T>
__global__ __launch_bounds__(SLICING_THREADS) void _reconstruct(T *__restrict__ dst,
                                                                const __grid_constant__ SlicedTensorView src,
                                                                int32_t rR, size_t n) {
  using Acc = T;
  __shared__ SliceDescriptor info;
  src.load_info(info);
  __syncthreads();

  const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) {
    return;
  }
  // Sum digits relative to rH, then apply rH once to avoid overflowing top-slice terms.
  Acc result = static_cast<Acc>(0.0);
  const int32_t planes = info.rH - info.rL + 1;
  for (int32_t s = 0; s < min(rR, planes); s++) {
    result += core::RealTraits<Acc>::up_cast(*src.ptr_from_linear(planes - s, i), core::SLICE_STEP * (s + 1));
  }
  dst[i] = static_cast<T>(core::RealTraits<Acc>::scalbn(result, core::SLICE_STEP * info.rH));
}

} // namespace ozaki::gpu

#endif // OZAKI_GPU_SLICING_KERNELS_CUH
