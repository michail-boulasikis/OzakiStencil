#ifndef HEADER_OZCONCEPTS_HPP
#define HEADER_OZCONCEPTS_HPP

#include <cmath>
#include <concepts>
#include <cstdint>

namespace ozi {
// forward declarations because fundamental types do not trigger ADL...
template <typename T> int32_t ozaki_scaling_factor(int32_t);
void matmul(int32_t *C, const int8_t *A, const int8_t *B, size_t m, size_t n, size_t k);

namespace detail {
using std::abs;
using std::frexp;
using std::max;
using std::scalbn;
using std::isfinite;

template <typename T>
concept HasNumericFunctions = requires(T a, T b, int32_t n) {
  { scalbn(a, n) } -> std::convertible_to<T>;
  { max(a, b) } -> std::convertible_to<T>;
  { abs(a) } -> std::convertible_to<T>;
  { frexp(a, &n) } -> std::convertible_to<T>;
  { isfinite(a) } -> std::convertible_to<bool>;
};
} // namespace detail

template <typename T>
concept OzakiData = detail::HasNumericFunctions<T> && requires(T a, T b) {
  { a + b } -> std::same_as<T>;
  { a - b } -> std::same_as<T>;
  { a * b } -> std::same_as<T>;
};

template <typename T>
concept OzakiTarget = requires(int32_t dyn_range) {
  // This now works because ozaki_scaling_factor is forward declared above
  { ozaki_scaling_factor<T>(dyn_range) } -> std::convertible_to<int32_t>;
};

// forward declarations because fundamental types do not trigger ADL...
template <OzakiData T, OzakiTarget U> U down_cast(T a, int32_t scale);
template <OzakiData T, OzakiTarget U> T up_cast(U a, int32_t scale);
template <OzakiData T, OzakiTarget U, typename A> T up_cast_acc(A a, int32_t scale, int32_t n_mul);

template <typename T, typename Accumulator>
concept OzakiMatMul = OzakiTarget<T> && requires(Accumulator *C, const T *A, const T *B, size_t m, size_t n, size_t k) {
  { matmul(C, A, B, m, n, k) } -> std::same_as<void>;
};

template <typename T>
concept OzakiPair =
    requires {
      typename T::data_type;
      typename T::target_type;
      typename T::acc_type;
    } && OzakiData<typename T::data_type> && OzakiTarget<typename T::target_type> &&
    OzakiMatMul<typename T::target_type, typename T::acc_type> &&
    requires(typename T::data_type d, typename T::target_type t, int32_t scale, int32_t n_mult) {
      { down_cast<typename T::data_type, typename T::target_type>(d, scale) } -> std::same_as<typename T::target_type>;
      { up_cast<typename T::data_type, typename T::target_type>(t, scale) } -> std::same_as<typename T::data_type>;
      { up_cast_acc<typename T::data_type, typename T::target_type, typename T::acc_type>(t, scale, n_mult) } -> std::same_as<typename T::data_type>;
    } && (sizeof(typename T::data_type) > sizeof(typename T::target_type));
} // namespace ozi

#endif // HEADER_OZCONCEPTS_HPP
