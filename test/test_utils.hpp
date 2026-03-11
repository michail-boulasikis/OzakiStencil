#pragma once

#include <ozaki/core/stencil.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace test_utils {

/**
 * Relative RMSE over interior cells only.
 * Interior cells are those at least offsets[d] away from each edge in dimension d.
 */
template <typename T, size_t NDims>
double interior_relative_rmse(const std::vector<T> &actual, const std::vector<T> &expected,
                              const std::array<size_t, NDims> &shape,
                              const std::array<size_t, NDims> &offsets) {
  size_t total = 1;
  for (auto s : shape)
    total *= s;

  double sum_sq_err = 0.0;
  double sum_sq_ref = 0.0;
  size_t count = 0;

  for (size_t lin = 0; lin < total; ++lin) {
    auto idx = ozi::multi_from_linear(lin, shape);
    bool interior = true;
    for (size_t d = 0; d < NDims; ++d) {
      if (idx[d] < offsets[d] || idx[d] >= shape[d] - offsets[d]) {
        interior = false;
        break;
      }
    }
    if (!interior)
      continue;

    double a = static_cast<double>(actual[lin]);
    double e = static_cast<double>(expected[lin]);
    sum_sq_err += (a - e) * (a - e);
    sum_sq_ref += e * e;
    count++;
  }

  if (count == 0)
    return 0.0;
  if (sum_sq_ref == 0.0)
    return std::sqrt(sum_sq_err / static_cast<double>(count));
  return std::sqrt(sum_sq_err / sum_sq_ref);
}

/**
 * Relative RMSE over boundary cells only.
 * Boundary cells are those within offsets[d] of any edge.
 */
template <typename T, size_t NDims>
double boundary_relative_rmse(const std::vector<T> &actual, const std::vector<T> &expected,
                              const std::array<size_t, NDims> &shape,
                              const std::array<size_t, NDims> &offsets) {
  size_t total = 1;
  for (auto s : shape)
    total *= s;

  double sum_sq_err = 0.0;
  double sum_sq_ref = 0.0;
  size_t count = 0;

  for (size_t lin = 0; lin < total; ++lin) {
    auto idx = ozi::multi_from_linear(lin, shape);
    bool interior = true;
    for (size_t d = 0; d < NDims; ++d) {
      if (idx[d] < offsets[d] || idx[d] >= shape[d] - offsets[d]) {
        interior = false;
        break;
      }
    }
    if (interior)
      continue;

    double a = static_cast<double>(actual[lin]);
    double e = static_cast<double>(expected[lin]);
    sum_sq_err += (a - e) * (a - e);
    sum_sq_ref += e * e;
    count++;
  }

  if (count == 0)
    return 0.0;
  if (sum_sq_ref == 0.0)
    return std::sqrt(sum_sq_err / static_cast<double>(count));
  return std::sqrt(sum_sq_err / sum_sq_ref);
}

/**
 * Multi-step gold standard: applies stencil T times in __float128 precision.
 * Boundary cells are zero-filled (apply_to skips them).
 */
template <typename T, size_t... Dims>
std::vector<T> compute_gold_standard_multistep(const ozi::Stencil<T, Dims...> &stencil,
                                               const std::vector<T> &input,
                                               const std::array<size_t, sizeof...(Dims)> &shape,
                                               int32_t timesteps) {
#ifdef __GNUC__
  using HighPrec = __float128;
#else
  using HighPrec = long double;
#endif

  ozi::Stencil<HighPrec, Dims...> s_high;
  s_high.offsets = stencil.offsets;
  for (size_t i = 0; i < stencil.coeffs.size(); ++i)
    s_high.coeffs[i] = static_cast<HighPrec>(stencil.coeffs[i]);

  size_t n = input.size();
  std::vector<HighPrec> buf_a(input.begin(), input.end());
  std::vector<HighPrec> buf_b(n);

  for (int32_t t = 0; t < timesteps; ++t) {
    // Copy buf_a to buf_b so boundary cells are preserved (matches GPU
    // behavior where emulated_gain keeps boundary cells ~= constant).
    // apply_to only overwrites interior cells.
    buf_b = buf_a;
    s_high.template apply_to<HighPrec>(buf_b, buf_a, shape);
    std::swap(buf_a, buf_b);
  }

  std::vector<T> result(n);
  for (size_t i = 0; i < n; ++i)
    result[i] = static_cast<T>(buf_a[i]);
  return result;
}

/**
 * Computes a gold standard result by performing stencil operations in high precision.
 * Uses __float128 on GCC/Clang when available, otherwise falls back to long double.
 *
 * @tparam T The output type
 * @tparam InputContainer The container type for input data
 * @tparam Dims The stencil dimensions
 * @param stencil The stencil to apply
 * @param input The input data
 * @param shape The shape of the data
 * @return A vector containing the gold standard result
 */
template <typename T, typename InputContainer, size_t... Dims>
std::vector<T> compute_gold_standard(const ozi::Stencil<T, Dims...> &stencil, const InputContainer &input,
                                     const std::array<size_t, sizeof...(Dims)> &shape) {

#ifdef __GNUC__
  using HighPrec = __float128;
#else
  using HighPrec = long double;
#endif

  ozi::Stencil<HighPrec, Dims...> s_high;

  s_high.offsets = stencil.offsets;
  for (size_t i = 0; i < stencil.coeffs.size(); ++i) {
    s_high.coeffs[i] = static_cast<HighPrec>(stencil.coeffs[i]);
  }

  std::vector<HighPrec> in_high(input.begin(), input.end());
  std::vector<HighPrec> out_high(input.size());

  s_high.template apply_to<HighPrec>(out_high, in_high, shape);

  std::vector<T> result(out_high.size());
  for (size_t i = 0; i < out_high.size(); ++i) {
    result[i] = static_cast<T>(out_high[i]);
  }

  return result;
}

} // namespace test_utils
