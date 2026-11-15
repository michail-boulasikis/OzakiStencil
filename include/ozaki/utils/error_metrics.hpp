#ifndef OZAKI_UTILS_ERROR_METRICS_HPP
#define OZAKI_UTILS_ERROR_METRICS_HPP

#include <quadmath.h>

#include <stdexcept>
#include <vector>

// Root-mean-square difference, computed in binary128.
template <typename T_Test, typename T_Ref>
__float128 calculate_rmse(const std::vector<T_Test> &test, const std::vector<T_Ref> &ref) {
  if (test.size() != ref.size()) {
    throw std::runtime_error("Size mismatch in RMSE calculation");
  }
  __float128 sum = 0;
  for (size_t i = 0; i < test.size(); ++i) {
    const __float128 diff = static_cast<__float128>(test[i]) - static_cast<__float128>(ref[i]);
    sum += diff * diff;
  }
  return sqrtq(sum / static_cast<__float128>(test.size()));
}

// Largest absolute difference, computed in binary128.
template <typename T_Test, typename T_Ref>
__float128 calculate_max_abs_diff(const std::vector<T_Test> &test, const std::vector<T_Ref> &ref) {
  if (test.size() != ref.size()) {
    throw std::runtime_error("Size mismatch in maximum absolute difference calculation");
  }
  __float128 max_diff = 0;
  for (size_t i = 0; i < test.size(); ++i) {
    max_diff = fmaxq(max_diff, fabsq(static_cast<__float128>(test[i]) - static_cast<__float128>(ref[i])));
  }
  return max_diff;
}

#endif // OZAKI_UTILS_ERROR_METRICS_HPP
