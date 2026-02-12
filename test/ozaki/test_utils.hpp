#pragma once

#include <ozaki/core/stencil.hpp>

#include <array>
#include <vector>

namespace test_utils {

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
