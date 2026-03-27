#ifndef HEADER_OZI_CUH
#define HEADER_OZI_CUH

#include "../core/oztraits.hpp"
#include "../cpu/ozi.hpp"

#include <climits>
#include <concepts>
#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 1000
#include <crt/device_fp128_functions.h>
#endif
#endif
#include <cuda_runtime.h>
#include <limits>
#include <type_traits>

#include "gpu_checks.cuh"

// Optional high-precision arithmetic for scale factor computation
// Define OZAKI_USE_BOOST_MULTIPRECISION to enable (requires Boost headers)
#ifdef OZAKI_USE_BOOST_MULTIPRECISION
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <string>
#ifdef OZAKI_HAVE_FLOAT128
#include <quadmath.h>
#endif
#endif

namespace ozigpu {

template <typename T>
  requires(sizeof(T) > 1)
struct OzakiFP_Int8 {
  using data_type = T;
  using target_type = int8_t;
  using acc_type = int32_t;
};

template <ozi::OzakiPair O> struct alignas(256) OzakiDescriptor {
  using Data = typename O::data_type;
  using Target = typename O::target_type;
  using Acc = typename O::acc_type;

  int32_t e_max = std::numeric_limits<int32_t>::min();
  int32_t e_min = std::numeric_limits<int32_t>::max();
  int32_t n_slices_input = 8;

  int32_t n_shifts = 0;
  int32_t shift_required = 0;
  int32_t just_shifted = 1;
  Target overflow_danger_threshold = std::numeric_limits<Target>::max() / 2;
  int32_t carry_slice_index = 0;
};

// Vectorized load/store helper
template <typename T> struct FourValues {
  T a[4];
};
} // namespace ozigpu

namespace ozi {
template <std::floating_point T> struct OzakiPairTraits<ozigpu::OzakiFP_Int8<T>> {
  using Data = typename ozigpu::OzakiFP_Int8<T>::data_type;
  using Target = typename ozigpu::OzakiFP_Int8<T>::target_type;
  using Acc = typename ozigpu::OzakiFP_Int8<T>::acc_type;

  __device__ __forceinline__ static Target down_cast(Data a, int32_t scale) {
    if constexpr (std::is_same_v<Data, double>) {
      double a_scaled = scalbn(a, scale);
      int32_t temp;
      asm("cvt.rni.sat.s8.f64 %0, %1;" : "=r"(temp) : "d"(a_scaled));
      return static_cast<Target>(temp);
    } else if constexpr (std::is_same_v<Data, float>) {
      float a_scaled = scalbn(static_cast<float>(a), scale);
      int32_t temp;
      asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(temp) : "f"(a_scaled));
      return static_cast<Target>(temp);
    } else {
      static_assert(!std::is_same_v<Data, Data>, "down_cast not implemented for this type yet.");
    }
  }

  __host__ __device__ __forceinline__ static Data up_cast(Target a, int32_t scale) {
    return OzakiTraits<Data>::scalbn(static_cast<Data>(a), -scale);
  }

  __host__ __device__ __forceinline__ static Data up_cast_acc(Acc a, int32_t scale, int32_t n_mult) {
    return OzakiTraits<Data>::scalbn(static_cast<Data>(a), -(n_mult * (CHAR_BIT * sizeof(Target) - 1) - scale));
  }

  static void matmul(Acc *C, const Target *A, const Target *B, size_t m, size_t n, size_t k) {
    // Not implemented for stencil operations
  }
};

#ifdef OZAKI_HAVE_FLOAT128
template <> struct OzakiPairTraits<ozigpu::OzakiFP_Int8<__float128>> {
  using Data = typename ozigpu::OzakiFP_Int8<__float128>::data_type;
  using Target = typename ozigpu::OzakiFP_Int8<__float128>::target_type;
  using Acc = typename ozigpu::OzakiFP_Int8<__float128>::acc_type;

  static __host__ __device__ __forceinline__ Target down_cast(Data a, int32_t scale) {
    __float128 a_scaled = OzakiTraits<Data>::scalbn(a, scale);
#ifdef __CUDA_ARCH__
    __float128 rounded = __nv_fp128_rint(a_scaled);
    // Saturate to int8_t range to prevent wraparound
    if (rounded > static_cast<__float128>(127))
      return 127;
    if (rounded < static_cast<__float128>(-128))
      return -128;
    return static_cast<Target>(rounded);
#else
    long long rounded = llrintq(a_scaled);
    // Saturate to int8_t range
    if (rounded > 127)
      return 127;
    if (rounded < -128)
      return -128;
    return static_cast<Target>(rounded);
#endif
  }

  static __host__ __device__ __forceinline__ Data up_cast(Target a, int32_t scale) {
    return OzakiTraits<Data>::scalbn(static_cast<Data>(a), -scale);
  }

  static __host__ __device__ __forceinline__ Data up_cast_acc(Acc a, int32_t scale, int32_t n_mult) {
    return OzakiTraits<Data>::scalbn(static_cast<Data>(a), -(n_mult * (CHAR_BIT * sizeof(Target) - 1) - scale));
  }

  static __host__ __device__ __forceinline__ void matmul(Acc *A, const Target *B, const Target *C, size_t m, size_t n,
                                                         size_t k) {} // not implemented; satisfies OzakiPair concept
};
#endif

} // namespace ozi

namespace ozigpu {
using namespace ozi;

/** Helper to shuffle down any type, including __float128 which requires component-wise shuffling */
template <typename T>
__device__ __forceinline__ T shfl_down_sync_wrapper(uint32_t mask, T val, uint32_t offset, int32_t width = 32) {
  if constexpr (sizeof(T) <= 8) {
    // Native shuffle support for types up to 64 bits
    return __shfl_down_sync(mask, val, offset, width);
#ifdef OZAKI_HAVE_FLOAT128
  } else if constexpr (std::is_same_v<T, __float128>) {
    // For __float128, split into two 64-bit components, shuffle each, then reconstruct
    RawInt<T> raw = bit_cast<RawInt<T>>(val);
    raw.hi = __shfl_down_sync(mask, raw.hi, offset, width);
    raw.lo = __shfl_down_sync(mask, raw.lo, offset, width);
    return bit_cast<T>(raw);
#endif
  } else {
    static_assert(sizeof(T) <= 16, "Unsupported type size for warp shuffle");
  }
}

/** Helper to compute ceil(log2(|val|)) for dynamic range analysis using frexp */
template <typename T> __device__ __forceinline__ int32_t log2_ceil_abs(T val) {
  T abs_val = OzakiTraits<T>::abs(val);
  if (abs_val == static_cast<T>(0)) {
    return std::numeric_limits<int32_t>::min();
  }
  int32_t exp;
  T frac = OzakiTraits<T>::frexp(abs_val, &exp);
  return (frac == static_cast<T>(0.5)) ? (exp - 1) : exp;
}

template <ozi::OzakiPair O>
__host__
#ifdef OZAKI_HAVE_FLOAT128
    __float128
#else
    long double
#endif
    total_integration_scale_factor_quad(HighPrecType a, int64_t time_steps, int64_t shifts) {
  using Target = typename O::target_type;
  constexpr int32_t slice_bits = CHAR_BIT * sizeof(Target) - 1;

#ifdef OZAKI_USE_BOOST_MULTIPRECISION
  using high_precision_float = boost::multiprecision::number<boost::multiprecision::backends::cpp_bin_float<512>>;

  // Convert to boost with maximum available precision.
#ifdef OZAKI_HAVE_FLOAT128
  // __float128 → boost via string to preserve full 113-bit mantissa.
  // static_cast<double>(a) would truncate to 53 bits, introducing ~2^-53
  // relative error that compounds through pow(a, -N).
  char a_buf[64];
  quadmath_snprintf(a_buf, sizeof(a_buf), "%.40Qe", static_cast<__float128>(a));
  high_precision_float a_hp(a_buf);
#else
  high_precision_float a_hp = static_cast<double>(a);
#endif
  high_precision_float result = boost::multiprecision::pow(a_hp, -time_steps);

  // Multiply by 2^(slice_bits * shifts)
  int64_t exp_shift = static_cast<int64_t>(slice_bits) * shifts;
  result *= boost::multiprecision::pow(high_precision_float(2), exp_shift);

#ifdef OZAKI_HAVE_FLOAT128
  std::string str = result.str(128);
  return strtoflt128(str.c_str(), nullptr);
#else
  return result.convert_to<long double>();
#endif

#else
  // Fallback: original implementation using native quad precision
  // Note: This has reduced accuracy for non-power-of-2 scaling factors

#ifdef OZAKI_HAVE_FLOAT128
  // We compute: 2^(slice_bits * shifts) / a^time_steps
  int exp;

  __float128 frac = frexpq(a, &exp);

  __float128 pow_frac = powq(frac, static_cast<__float128>(time_steps));

  __float128 inv_pow_frac = static_cast<__float128>(1.0) / pow_frac;

  // Result = 2^(slice_bits * shifts) / (frac^time_steps * 2^(exp * time_steps))
  //        = (1 / frac^time_steps) * 2^(slice_bits * shifts - exp * time_steps)
  int64_t total_exp = static_cast<int64_t>(slice_bits) * shifts - static_cast<int64_t>(exp) * time_steps;

  return scalbnq(inv_pow_frac, static_cast<int>(total_exp));

#else
  int exp;

  long double frac = std::frexp(a, &exp);
  long double pow_frac = std::pow(frac, static_cast<long double>(time_steps));

  long double inv_pow_frac = static_cast<long double>(1.0) / pow_frac;

  int64_t total_exp = static_cast<int64_t>(slice_bits) * shifts - static_cast<int64_t>(exp) * time_steps;

  return std::scalbn(inv_pow_frac, static_cast<int>(total_exp));
#endif
#endif
}
} // namespace ozigpu

#endif // HEADER_OZI_CUH
