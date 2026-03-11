#ifndef HEADER_OZTRAITS_HPP
#define HEADER_OZTRAITS_HPP

#include <bit>
#include <cstring>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <crt/device_fp128_functions.h>
#include <cuda_fp16.h>
#define HDF __host__ __device__ __forceinline__
#else
#define HDF
#endif

#ifdef __GNUC__
#include <quadmath.h>
#endif

namespace ozi {

template <typename T> struct OzakiTraits {};
template <typename T> struct OzakiPairTraits {};

template <size_t Bytes> struct UintByType;
template <> struct UintByType<2> {
  using type = uint16_t;
};
template <> struct UintByType<4> {
  using type = uint32_t;
};
template <> struct UintByType<8> {
  using type = uint64_t;
};
template <> struct UintByType<16> {
  using type = struct {
    uint64_t lo;
    uint64_t hi;
  };
};

template <typename T> using RawInt = typename UintByType<sizeof(T)>::type;

template <typename To, typename From> HDF To bit_cast(const From &src) {
  using std::memcpy;
  static_assert(sizeof(To) == sizeof(From), "Size mismatch in bit_cast");
  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename T> struct FloatTraits;
#ifdef __CUDA_ARCH__
template <> struct FloatTraits<__half> {
  static constexpr int32_t MantissaBits = 10;
  static constexpr int32_t ExponentBias = 15;
};
#endif
template <> struct FloatTraits<float> {
  static constexpr int32_t MantissaBits = 23;
  static constexpr int32_t ExponentBias = 127;
};
template <> struct FloatTraits<double> {
  static constexpr int32_t MantissaBits = 52;
  static constexpr int32_t ExponentBias = 1023;
};
#if __GNUC__ || __CUDA_ARCH__ >= 100
template <> struct FloatTraits<__float128> {
  static constexpr int32_t MantissaBits = 112;
  static constexpr int32_t ExponentBias = 16383;
};
#endif

template <typename T> HDF int32_t ffs(T val) {
#ifdef __CUDA_ARCH__
  if constexpr (sizeof(T) <= 4) {
    return __ffs(static_cast<int32_t>(val));
  } else if constexpr (sizeof(T) <= 8) {
    return __ffsll(static_cast<int64_t>(val));
  } else {
    static_assert(sizeof(T) == 0, "`ffs` not supported for types with size greater than 8 bytes");
  }
#else
  return std::countr_zero(val) + 1;
#endif
}

template <typename T> HDF int32_t dyn_min_lsb(T a, int32_t b) {
  static_assert(sizeof(T) == 0, "`dyn_min_lsb` is not implemented for this type.");
  return -1;
}

template <std::floating_point T> HDF int32_t dyn_min_lsb(T a, int32_t current_lsb) {
  using IntType = RawInt<T>;
  using Traits = FloatTraits<T>;
  using std::min;

  if (a == static_cast<T>(0)) // zero does not count.
    return current_lsb;

  IntType raw = bit_cast<IntType>(a);

  constexpr IntType One = 1;
  constexpr int32_t M_Bits = Traits::MantissaBits;
  constexpr IntType M_Mask = (One << M_Bits) - 1;
  constexpr int32_t TotalBits = sizeof(T) * CHAR_BIT;
  IntType exponent_bits = (raw >> M_Bits) & ((One << (TotalBits - 1 - M_Bits)) - 1);
  IntType mantissa = raw & M_Mask;
  if (exponent_bits != 0) {
    mantissa |= (One << M_Bits);
  } else {
    exponent_bits = 1;
  }
  int32_t ffs_index = ffs(mantissa);
  int32_t bit_exponent = (static_cast<int32_t>(exponent_bits) - Traits::ExponentBias) - (M_Bits - (ffs_index - 1));
  return min(current_lsb, bit_exponent);
}

#if __GNUC__ || __CUDA_ARCH__ >= 100
template <> HDF int32_t dyn_min_lsb(__float128 a, int32_t current_lsb) {
  using IntType = RawInt<__float128>;
  using Traits = FloatTraits<__float128>;
  using std::min;

  if (a == static_cast<__float128>(0))
    return current_lsb;

  IntType raw = bit_cast<IntType>(a);

  constexpr int32_t M_Bits_High = 48;
  constexpr uint64_t One = 1;
  constexpr uint64_t M_Mask_High = (One << M_Bits_High) - 1;
  constexpr uint64_t Exp_Mask = (One << 15) - 1;

  uint64_t mantissa_high = raw.hi & M_Mask_High;
  uint64_t exponent = (raw.hi >> M_Bits_High) & Exp_Mask;

  if (exponent != 0) {
    mantissa_high |= (One << M_Bits_High);
  } else {
    exponent = 1;
  }

  int32_t global_ffs_index;

  int32_t ffs_low = ffs(raw.lo);

  if (ffs_low != 0) {
    global_ffs_index = ffs_low;
  } else {
    int32_t ffs_high = ffs(mantissa_high);
    global_ffs_index = 64 + ffs_high;
  }

  int32_t bit_exponent =
      (static_cast<int32_t>(exponent) - Traits::ExponentBias) - (Traits::MantissaBits - (global_ffs_index - 1));

  return min(bit_exponent, current_lsb);
}
#endif

template <std::floating_point T> struct OzakiTraits<T> {
  static HDF T abs(T a) {
    using std::abs;
    return abs(a);
  }
  static HDF T max(T a, T b) {
    using std::max;
    return max(a, b);
  }
  static HDF T scalbn(T a, int n) {
    using std::scalbn;
    return scalbn(a, n);
  }
  static HDF T frexp(T a, int *n) {
    using std::frexp;
    return frexp(a, n);
  }
  static HDF bool isfinite(T a) {
    using std::isfinite;
    return isfinite(a);
  }
};

template <std::integral T> struct OzakiTraits<T> {
  static HDF int32_t scaling_factor(int32_t v) {
    return (CHAR_BIT * sizeof(T) - 1) - v;
  }
  static HDF int32_t slice_step(int32_t v) { return v - (CHAR_BIT * sizeof(T) - 1); }
};

#if __GNUC__ || __CUDA_ARCH__ >= 100
template <> struct OzakiTraits<__float128> {
  static HDF __float128 abs(__float128 a) {
#ifdef __CUDA_ARCH__
    return __nv_fp128_fabs(a);
#else
    return fabsq(a);
#endif
  }
  static HDF __float128 max(__float128 a, __float128 b) {
#ifdef __CUDA_ARCH__
    return __nv_fp128_fmax(a, b);
#else
    return fmaxq(a, b);
#endif
  }
  static HDF __float128 frexp(__float128 a, int *n) {
#ifdef __CUDA_ARCH__
    return __nv_fp128_frexp(a, n);
#else
    return frexpq(a, n);
#endif
  }
  static HDF bool isfinite(__float128 a) { 
#ifdef __CUDA_ARCH__
    return true; // unused in CUDA
#else
    return (!isnanq(a) && !isinfq(a));
#endif
  }
  static HDF __float128 scalbn(__float128 a, int n) {
#ifdef __CUDA_ARCH__
    using IntType = RawInt<__float128>;
    using Traits = FloatTraits<__float128>;

    // 2^114: exponent = 16383 + 114 = 16497 = 0x4071
    IntType Two114_bits;
    Two114_bits.lo = 0;
    Two114_bits.hi = 0x4071000000000000ULL;
    const __float128 Two114 = bit_cast<__float128>(Two114_bits);

    // 2^-114: exponent = 16383 - 114 = 16269 = 0x3F8D
    IntType Twom114_bits;
    Twom114_bits.lo = 0;
    Twom114_bits.hi = 0x3F8D000000000000ULL;
    const __float128 Twom114 = bit_cast<__float128>(Twom114_bits);

    IntType u = bit_cast<IntType>(a);
    int64_t k = (u.hi >> 48) & 0x7FFF;

    const uint64_t original_sign = u.hi & 0x8000000000000000ULL;

    if (k == 0) {
      if ((u.hi & 0x0000FFFFFFFFFFFFULL) == 0 && u.lo == 0) {
        return a;
      }
      a *= Two114;
      u = bit_cast<IntType, __float128>(a);
      k = ((u.hi >> 48) & 0x7FFF) - 114;
    }

    if (k == 0x7FFF) {
      return a;
    }

    if (n > 35000)
      n = 35000;
    if (n < -35000)
      n = -35000;

    k = k + n;

    if (k > 0 && k < 0x7FFF) {
      // Construct result with original sign, new exponent, preserved mantissa
      uint64_t mantissa_bits = u.hi & 0x0000FFFFFFFFFFFFULL;  // Extract mantissa (bits 47-0)
      uint64_t exponent_bits = (static_cast<uint64_t>(k) & 0x7FFFULL) << 48;  // Exponent at bits 62-48
      IntType result;
      result.lo = u.lo;
      result.hi = original_sign | exponent_bits | mantissa_bits;
      return bit_cast<__float128, IntType>(result);
    }

    if (k >= 0x7FFF) {
      IntType result;
      result.lo = 0;
      result.hi = original_sign | 0x7FFF000000000000ULL;
      return bit_cast<__float128, IntType>(result);
    }

    if (k <= 0) {
      if (k > -114) {
        k += 114;
        uint64_t mantissa_bits = u.hi & 0x0000FFFFFFFFFFFFULL;
        uint64_t exponent_bits = (static_cast<uint64_t>(k) & 0x7FFFULL) << 48;
        IntType temp;
        temp.lo = u.lo;
        temp.hi = original_sign | exponent_bits | mantissa_bits;
        __float128 temp_normal = bit_cast<__float128, IntType>(temp);

        return temp_normal * Twom114; // Software multiply handles the denormal shift/round
      } else {
        IntType result;
        result.lo = 0;
        result.hi = original_sign;
        return bit_cast<__float128, IntType>(result);
      }
    }

    return a;
#else
    return scalbnq(a, n);
#endif
  }
};
#endif

template <typename T>
concept OzakiData = requires(T a, T b, int32_t n) {
  { OzakiTraits<T>::scalbn(a, n) } -> std::convertible_to<T>;
  { OzakiTraits<T>::max(a, b) } -> std::convertible_to<T>;
  { OzakiTraits<T>::abs(a) } -> std::convertible_to<T>;
  { OzakiTraits<T>::frexp(a, &n) } -> std::convertible_to<T>;
  { OzakiTraits<T>::isfinite(a) } -> std::convertible_to<bool>;
  { a + b } -> std::same_as<T>;
  { a - b } -> std::same_as<T>;
  { a * b } -> std::same_as<T>;
};

template <typename T>
concept OzakiTarget = requires(int32_t dyn_range) {
  { OzakiTraits<T>::scaling_factor(dyn_range) } -> std::convertible_to<int32_t>;
  { OzakiTraits<T>::slice_step(dyn_range) } -> std::convertible_to<int32_t>;
};


template <typename T>
concept OzakiPair =
    requires {
      typename T::data_type;
      typename T::target_type;
      typename T::acc_type;
    } && OzakiData<typename T::data_type> && OzakiTarget<typename T::target_type> &&
    requires(typename T::data_type d, typename T::target_type t, typename T::acc_type a, int32_t scale, int32_t n_mult) {
      { OzakiPairTraits<T>::down_cast(d, scale) } -> std::same_as<typename T::target_type>;
      { OzakiPairTraits<T>::up_cast(t, scale) } -> std::same_as<typename T::data_type>;
      { OzakiPairTraits<T>::up_cast_acc(a, scale, n_mult) } -> std::same_as<typename T::data_type>;
    } && requires (typename T::acc_type *C, const typename T::target_type *A, const typename T::target_type *B, size_t m, size_t n, size_t k) {
      { OzakiPairTraits<T>::matmul(C, A, B, m, n, k) } -> std::same_as<void>;
    } && (sizeof(typename T::data_type) > sizeof(typename T::target_type));

} // namespace ozi

#endif // HEADER_OZTRAITS_HPP
