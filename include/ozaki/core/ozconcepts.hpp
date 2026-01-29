#ifndef HEADER_OZCONCEPTS_HPP
#define HEADER_OZCONCEPTS_HPP

#include <cmath>
#include <concepts>
#include <cstdint>

namespace ozi {

template <typename T> struct OzakiTraits {};
template <typename T> struct OzakiPairTraits {};

#ifdef __CUDACC__
#define HDF __host__ __device__ __forceinline__
#else
#define HDF
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
    requires(typename T::data_type d, typename T::target_type t, int32_t scale, int32_t n_mult) {
      { OzakiPairTraits<T>::down_cast(d, scale) } -> std::same_as<typename T::target_type>;
      { OzakiPairTraits<T>::up_cast(t, scale) } -> std::same_as<typename T::data_type>;
      { OzakiPairTraits<T>::up_cast_acc(t, scale, n_mult) } -> std::same_as<typename T::data_type>;
    } && requires (typename T::acc_type *C, const typename T::target_type *A, const typename T::target_type *B, size_t m, size_t n, size_t k) {
      { OzakiPairTraits<T>::matmul(C, A, B, m, n, k) } -> std::same_as<void>;
    } && (sizeof(typename T::data_type) > sizeof(typename T::target_type));
} // namespace ozi

#endif // HEADER_OZCONCEPTS_HPP
