// Floating-point operations shared by host and device code for the two value
// types OzakiStencil integrates in: double and __float128 (device binary128
// exists on SM100 and newer).
#ifndef OZAKI_CORE_TRAITS_HPP
#define OZAKI_CORE_TRAITS_HPP

#include <climits>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#include <crt/device_fp128_functions.h>
#endif
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_INLINE __forceinline__
#else
#define CUDA_HOST
#define CUDA_DEVICE
#define CUDA_INLINE
#endif

#ifndef __CUDA_ARCH__
#include <quadmath.h>
#endif

namespace core {

// Slices are int8 digits in radix 2^SLICE_STEP = 128.
using EmulationType = int8_t;
constexpr int32_t SLICE_STEP = sizeof(EmulationType) * CHAR_BIT - 1;

template <typename T> struct RealTraits;

template <> struct RealTraits<double> {
  static CUDA_HOST CUDA_DEVICE CUDA_INLINE double abs(double a) { return std::fabs(a); }
  static CUDA_HOST CUDA_DEVICE CUDA_INLINE double max(double a, double b) { return std::fmax(a, b); }
  static CUDA_HOST CUDA_DEVICE CUDA_INLINE double scalbn(double a, int n) { return std::scalbn(a, n); }
  static CUDA_HOST CUDA_DEVICE CUDA_INLINE double frexp(double a, int *n) { return std::frexp(a, n); }
  static CUDA_HOST CUDA_DEVICE CUDA_INLINE bool isfinite(double a) { return std::isfinite(a); }
  // Rounds a * 2^n to the nearest int8 (saturating).
  static CUDA_HOST CUDA_DEVICE CUDA_INLINE EmulationType down_cast(double a, int32_t n) {
#ifdef __CUDA_ARCH__
    int32_t digit;
    asm("cvt.rni.sat.s8.f64 %0, %1;" : "=r"(digit) : "d"(scalbn(a, n)));
    return static_cast<EmulationType>(digit);
#else
    const long long rounded = std::llrint(scalbn(a, n));
    return static_cast<EmulationType>(rounded < -128 ? -128 : rounded > 127 ? 127 : rounded);
#endif
  }
  // Returns digit * 2^-n.
  static CUDA_HOST CUDA_DEVICE CUDA_INLINE double up_cast(EmulationType digit, int32_t n) {
    return scalbn(static_cast<double>(digit), -n);
  }
};

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 1000
template <> struct RealTraits<__float128> {
#ifdef __CUDA_ARCH__
  static CUDA_DEVICE CUDA_INLINE __float128 abs(__float128 a) { return __nv_fp128_fabs(a); }
  static CUDA_DEVICE CUDA_INLINE __float128 max(__float128 a, __float128 b) { return __nv_fp128_fmax(a, b); }
  static CUDA_DEVICE CUDA_INLINE __float128 scalbn(__float128 a, int n) { return __nv_fp128_ldexp(a, n); }
  static CUDA_DEVICE CUDA_INLINE __float128 frexp(__float128 a, int *n) { return __nv_fp128_frexp(a, n); }
  static CUDA_DEVICE CUDA_INLINE __float128 rint(__float128 a) { return __nv_fp128_rint(a); }
#else
  static __float128 abs(__float128 a) { return fabsq(a); }
  static __float128 max(__float128 a, __float128 b) { return fmaxq(a, b); }
  static __float128 scalbn(__float128 a, int n) { return scalbnq(a, n); }
  static __float128 frexp(__float128 a, int *n) { return frexpq(a, n); }
  static __float128 rint(__float128 a) { return rintq(a); }
#endif
  // a - a is NaN exactly for infinities and NaNs.
  static CUDA_HOST CUDA_DEVICE CUDA_INLINE bool isfinite(__float128 a) { return a - a == 0; }
  static CUDA_HOST CUDA_DEVICE CUDA_INLINE EmulationType down_cast(__float128 a, int32_t n) {
    const __float128 rounded = rint(scalbn(a, n));
    return static_cast<EmulationType>(rounded < -128 ? -128 : rounded > 127 ? 127 : static_cast<int32_t>(rounded));
  }
  static CUDA_HOST CUDA_DEVICE CUDA_INLINE __float128 up_cast(EmulationType digit, int32_t n) {
    return scalbn(static_cast<__float128>(digit), -n);
  }
};
#endif

template <typename T>
concept Real = requires(T a, int32_t n, EmulationType digit) {
  { RealTraits<T>::scalbn(a, n) } -> std::same_as<T>;
  { RealTraits<T>::frexp(a, &n) } -> std::same_as<T>;
  { RealTraits<T>::down_cast(a, n) } -> std::same_as<EmulationType>;
  { RealTraits<T>::up_cast(digit, n) } -> std::same_as<T>;
};

} // namespace core

#endif // OZAKI_CORE_TRAITS_HPP
