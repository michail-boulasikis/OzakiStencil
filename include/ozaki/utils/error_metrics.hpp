#ifndef HEADER_ERROR_METRICS_HPP
#define HEADER_ERROR_METRICS_HPP

#include <cmath>
#include <stdexcept>
#include <vector>

#ifndef OZAKI_HAVE_FLOAT128
#if (defined(__GNUC__) && !defined(__NVCOMPILER) && !defined(__CUDA_ARCH__)) || __CUDA_ARCH__ >= 1000
#define OZAKI_HAVE_FLOAT128 1
#endif
#endif

#ifdef OZAKI_HAVE_FLOAT128
#include <quadmath.h>
#define HIGH_PREC __float128
#define HIGH_PREC_MAX(X, Y) fmaxq(X, Y)
#define HIGH_PREC_ABS(X) fabsq(X)
#else
#define HIGH_PREC long double
#define HIGH_PREC_ABS(X) std::abs(X)
#define HIGH_PREC_MAX(X, Y) std::max(X, Y)
#endif

template <typename T_Test, typename T_Ref>
HIGH_PREC calculate_rmse(const std::vector<T_Test> &test, const std::vector<T_Ref> &ref) {
  if (test.size() != ref.size()) {
    throw std::runtime_error("Size mismatch in RMSE calculation");
  }
  HIGH_PREC err = static_cast<HIGH_PREC>(0.0);
  for (size_t i = 0; i < test.size(); ++i) {
    HIGH_PREC diff = static_cast<HIGH_PREC>(test[i]) - static_cast<HIGH_PREC>(ref[i]);
    err += diff * diff;
  }
#ifdef OZAKI_HAVE_FLOAT128
  return sqrtq(err / static_cast<HIGH_PREC>(test.size()));
#else
  return std::sqrt(err / static_cast<HIGH_PREC>(test.size()));
#endif
}

template <typename T_Test, typename T_Ref>
HIGH_PREC calculate_max_abs_diff(const std::vector<T_Test> &test, const std::vector<T_Ref> &ref) {
  if (test.size() != ref.size()) {
    throw std::runtime_error("Size mismatch in maximum absolute difference calculation");
  }
  HIGH_PREC max_diff = static_cast<HIGH_PREC>(0.0);
  for (size_t i = 0; i < test.size(); ++i) {
    HIGH_PREC diff = HIGH_PREC_ABS(static_cast<HIGH_PREC>(test[i]) - static_cast<HIGH_PREC>(ref[i]));
    max_diff = HIGH_PREC_MAX(max_diff, diff);
  }
  return max_diff;
}

#endif // HEADER_ERROR_METRICS_HPP
