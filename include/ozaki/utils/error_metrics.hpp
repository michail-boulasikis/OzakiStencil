#ifndef HEADER_ERROR_METRICS_HPP
#define HEADER_ERROR_METRICS_HPP

#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef __GNUC__
#include <quadmath.h>
#define HIGH_PREC __float128
#else
#define HIGH_PREC long double
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
#ifdef __GNUC__
  return sqrtq(err / static_cast<HIGH_PREC>(test.size()));
#else
  return std::sqrt(err / static_cast<HIGH_PREC>(test.size()));
#endif
}

#endif // HEADER_ERROR_METRICS_HPP
