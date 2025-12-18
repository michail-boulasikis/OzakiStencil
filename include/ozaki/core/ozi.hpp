#ifndef HEADER_OZAKI_HPP
#define HEADER_OZAKI_HPP

#include <climits>
#include <concepts>
#include <iostream>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

namespace ozaki {

template <typename T>
concept OzakiTarget = std::integral<T> && (sizeof(T) <= 2);
template <typename T>
concept OzakiData = std::floating_point<T>;

template <OzakiTarget U> HD static constexpr size_t effective_type_width() { return CHAR_BIT * sizeof(U) - 1; }
template <OzakiData T, OzakiTarget U> HD static constexpr size_t decomposition_size() {
  double decimal_size = static_cast<double>(CHAR_BIT * sizeof(T)) / effective_type_width<U>();
  size_t integer_size = static_cast<size_t>(decimal_size);
  return decimal_size > integer_size ? integer_size + 1 : integer_size;
}

template <OzakiData T, OzakiTarget U> HD static constexpr U round_and_clamp(T x) {
  constexpr auto lo = std::numeric_limits<U>::min() + 1;
  constexpr auto hi = std::numeric_limits<U>::max();
  auto r = std::llround(x);
  if (r < lo)
    r = lo;
  if (r > hi)
    r = hi;
  return static_cast<U>(r);
}

template <OzakiData T, OzakiTarget U> int32_t get_scaling_log2(const T amax) {
  int32_t e;
  std::frexp(amax, &e);
  return static_cast<int32_t>(effective_type_width<U>()) - e;
}

template <OzakiData T> T find_abs_max(const std::vector<T> &a) {
  T abs_max = static_cast<T>(0);
  const T *__restrict src = a.data();
  for (size_t i = 0; i < a.size(); ++i)
    abs_max = std::max(std::abs(src[i]), abs_max);
  return abs_max;
}

template <OzakiData T, OzakiTarget U>
void update_residual(std::vector<T> &res, std::vector<U> &slice, const int32_t exponent) {
  T *__restrict rptr = res.data();
  U *__restrict sptr = slice.data();
  for (size_t i = 0; i < res.size(); ++i) {
    sptr[i] = round_and_clamp<T, U>(std::ldexp(rptr[i], exponent));
    rptr[i] = rptr[i] - std::ldexp(static_cast<T>(sptr[i]), -exponent);
  }
}

template <OzakiData T, OzakiTarget U> struct OzakiISlice {
  int32_t exp;
  T scale; // precomputed scaling factor
  std::vector<U> data;

  OzakiISlice(int32_t exp, std::vector<U> data) : exp(exp), scale(std::ldexp(T(1), -exp)), data(std::move(data)) {}
  OzakiISlice() = default;

  std::vector<T> rescale() const {
    const size_t n = data.size();
    if (n == 0)
      return {};
    std::vector<T> out(n, 0);

    const U *__restrict src = data.data();
    T *__restrict dst = out.data();

    for (size_t i = 0; i < n; ++i)
      dst[i] = static_cast<T>(src[i]) * scale;

    return out;
  }

  void rescale_accumulate(std::vector<T> &out) const {
    const size_t n = std::min(out.size(), data.size());
    if (n == 0)
      return;
    const U *__restrict src = data.data();
    T *__restrict dst = out.data();

    for (size_t i = 0; i < n; ++i)
      dst[i] = std::fma(static_cast<T>(src[i]), scale, dst[i]);

    if (out.size() != data.size())
      std::cerr << "WARNING: accumulation using vectors of unequal size.\n";
  }
};

template <OzakiData T, OzakiTarget U> struct OzakiIMatrix {
  using data_type = T;
  using target_type = U;

  const size_t m, n;
  std::vector<OzakiISlice<T, U>> slices;

  OzakiIMatrix(const std::vector<T> &a, size_t lda) : m(lda), n(a.size() / lda), slices{} {
    auto residual = a;
    while (true) {
      T abs_max = find_abs_max(residual);

      if (!std::isfinite(abs_max))
        throw std::runtime_error("Ozaki scheme broke down or data included infinities/nans");
      if (abs_max == static_cast<T>(0)) {
        break;
      }

      int32_t exponent = get_scaling_log2<T, U>(abs_max);
      std::vector<U> slice_data(a.size(), 0);
      update_residual(residual, slice_data, exponent);
      slices.push_back(OzakiISlice<T, U>(exponent, slice_data));
    }
  }

  HD OzakiISlice<T, U> get_slice(size_t idx) const { return slices.at(idx); }

  std::vector<T> recombine() const {
    std::vector<T> res(m * n, 0);
    for (size_t s = slices.size(); s > 0; --s)
      slices[s - 1].rescale_accumulate(res);
    return res;
  }

  template <typename accumulator_type> std::vector<T> matmul(const OzakiIMatrix<T, U> &other) {
    if (this->n != other.m)
      throw std::runtime_error("Cannot multiply matrices of size m1xn1, m2xn2 where n1 != m2.");
    int32_t max_exponent_pair = std::numeric_limits<int32_t>::min();
    for (const auto &sliceA : this->slices)
      for (const auto &sliceB : other.slices)
        if (max_exponent_pair < -(sliceA.exp + sliceB.exp))
          max_exponent_pair = -(sliceA.exp + sliceB.exp);
    std::cout << "max_exponent_pair = " << max_exponent_pair << "\n";
    std::vector<T> res(m * n, static_cast<T>(0));
    for (const auto &sliceA : this->slices)
      for (const auto &sliceB : other.slices) {
        // optimization for later: check if accuracy will be affected by this matmul
        // if (f(sliceA.exp + sliceB.exp)) // maybe dependent on other exps too
        //    continue;
        std::cout << "\texponent_pair = " << -(sliceA.exp + sliceB.exp) << "\n";
        for (size_t row = 0; row < this->m; ++row)
          for (size_t col = 0; col < other.n; ++col) {
            accumulator_type sum = static_cast<accumulator_type>(0);
            for (size_t k = 0; k < this->n; ++k) {
              sum += static_cast<accumulator_type>(sliceA.data[row * this->n + k]) *
                     static_cast<accumulator_type>(sliceB.data[col + other.n * k]);
            }
            res[col + row * other.n] += std::ldexp(static_cast<T>(sum), -(sliceA.exp + sliceB.exp));
          }
      }
    return res;
  }
};

} // namespace ozaki

#endif // HEADER_OZAKI_HPP
