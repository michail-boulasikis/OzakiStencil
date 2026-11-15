#ifndef OZAKI_GPU_DIGIT_MATH_CUH
#define OZAKI_GPU_DIGIT_MATH_CUH

#include <ozaki/core/traits.hpp>
#include <ozaki/gpu/gpu_checks.cuh>

#include <cstdint>
#include <limits>

CUDA_HOST CUDA_DEVICE CUDA_INLINE constexpr int32_t cdiv(int32_t value, int32_t denom) {
  const int32_t quotient = value / denom;
  const int32_t remainder = value % denom;
  if (remainder != 0 && value > 0) {
    return quotient + 1;
  }
  return quotient;
}

CUDA_HOST CUDA_DEVICE CUDA_INLINE int32_t clz_u32(uint32_t value) {
#ifdef __CUDA_ARCH__
  return __clz(value);
#elif defined(__GNUC__) || defined(__clang__)
  return value == 0U ? 32 : __builtin_clz(value);
#else
  int32_t zeros = 0;
  for (uint32_t bit = 0x8000'0000U; bit != 0U && (value & bit) == 0U; bit >>= 1) {
    ++zeros;
  }
  return zeros;
#endif
}

CUDA_HOST CUDA_DEVICE CUDA_INLINE constexpr int64_t extract_carry(int64_t raw, int8_t digit) {
  return (raw - static_cast<int64_t>(digit)) >> core::SLICE_STEP;
}

CUDA_HOST CUDA_DEVICE CUDA_INLINE constexpr size_t align_up(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

namespace ozaki::gpu::detail {

// The largest diagonal contains nine complete 15x15 int8 products. Add the
// converged geometric carry bound (radix 128) so the interval also covers the
// carry entering the next diagonal.
constexpr int32_t MAX_INT8_STENCIL_MMA_SUM = 225 * 9 * 128 * 128;
constexpr int32_t MAX_INT8_STENCIL_ACCUMULATOR = MAX_INT8_STENCIL_MMA_SUM + (MAX_INT8_STENCIL_MMA_SUM + 126) / 127 + 2;
static_assert(MAX_INT8_STENCIL_ACCUMULATOR < std::numeric_limits<int32_t>::max() - 128);

// Carry out of an int32 accumulator whose low byte is kept as a signed int8
// digit: value == int8_t(value) + 128 * digit_carry(value). (Two radix units
// carry whenever the low byte crosses its signed midpoint.)
CUDA_HOST CUDA_DEVICE CUDA_INLINE constexpr int32_t digit_carry(int32_t value) {
  static_assert(core::SLICE_STEP == 7, "digit_carry assumes int8 slices with a 7-bit slice step.");
  return ((value >> 7) + 1) & ~1; // == 2 * ((value + 128) >> 8)
}

} // namespace ozaki::gpu::detail

#endif // OZAKI_GPU_DIGIT_MATH_CUH
