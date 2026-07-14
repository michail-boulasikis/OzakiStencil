#include <ozaki/utils/stencil_registry.hpp>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>

namespace ozi {

int32_t precision_mantissa_bits(OriginalPrecision p) {
  switch (p) {
  case OriginalPrecision::Int8:        return 7;
  case OriginalPrecision::Float:       return 24;
  case OriginalPrecision::Double:      return 53;
  case OriginalPrecision::LongDouble:  return 64;
  case OriginalPrecision::Float128:    return 113;
  }
  return 53;
}

OriginalPrecision detect_coefficient_precision(const StencilSpec &spec) {
#ifdef OZAKI_HAVE_FLOAT128
  bool fits_float  = true;
  bool fits_double = true;
  for (const auto &c : spec.coeffs) {
    if (c == CoeffStorageType(0)) continue;
    if (CoeffStorageType(static_cast<double>(c)) != c) { fits_double = false; fits_float = false; break; }
    if (CoeffStorageType(static_cast<float> (c)) != c)   fits_float  = false;
  }
  if (fits_float)  return OriginalPrecision::Float;
  if (fits_double) return OriginalPrecision::Double;
#endif
  return spec.specified_precision;
}

namespace detail {

CoeffStorageType coeff_abs(CoeffStorageType x) {
  return (x >= CoeffStorageType(0)) ? x : -x;
}

CoeffStorageType coeff_round(CoeffStorageType x) {
#ifdef OZAKI_HAVE_FLOAT128
  return roundq(x);
#else
  return std::round(x);
#endif
}

int32_t coeff_ilogb(CoeffStorageType x) {
#ifdef OZAKI_HAVE_FLOAT128
  return static_cast<int32_t>(ilogbq(x));
#else
  return std::ilogb(x);
#endif
}

CoeffStorageType coeff_scalbn(CoeffStorageType x, int32_t n) {
#ifdef OZAKI_HAVE_FLOAT128
  return scalbnq(x, n);
#else
  return std::scalbn(x, n);
#endif
}

} // namespace detail

StencilVariant make_stencil(const StencilSpec &spec) {
  auto try_make = [&]<typename T, size_t... Dims>(Stencil<T, Dims...> *) -> bool {
    constexpr std::array<size_t, sizeof...(Dims)> required_shape = {Dims...};
    if (spec.shape.size() != sizeof...(Dims))
      return false;
    for (size_t i = 0; i < sizeof...(Dims); ++i) {
      if (static_cast<size_t>(spec.shape[i]) != required_shape[i])
        return false;
    }
    return true;
  };

  auto populate = [&]<typename T, size_t... Dims>(Stencil<T, Dims...> &s) {
    for (size_t i = 0; i < sizeof...(Dims); ++i)
      s.offsets[i] = static_cast<size_t>(spec.offsets[i]);
    for (size_t i = 0; i < s.size; ++i)
      s.coeffs[i] = spec.coeffs[i];
  };

  StencilVariant result;
  bool found = false;

  auto try_variant = [&]<typename Alt>(Alt *tag) {
    if (!found && try_make(tag)) {
      Alt s;
      populate(s);
      result = s;
      found = true;
    }
  };

  std::apply([&](auto *...tags) { (try_variant(tags), ...); },
             std::tuple{
                 static_cast<Stencil<CoeffStorageType, 3> *>(nullptr),
                 static_cast<Stencil<CoeffStorageType, 5> *>(nullptr),
                 static_cast<Stencil<CoeffStorageType, 7> *>(nullptr),
                 static_cast<Stencil<CoeffStorageType, 3, 3> *>(nullptr),
                 static_cast<Stencil<CoeffStorageType, 5, 5> *>(nullptr),
                 static_cast<Stencil<CoeffStorageType, 7, 7> *>(nullptr),
                 static_cast<Stencil<CoeffStorageType, 3, 3, 3> *>(nullptr),
                 static_cast<Stencil<CoeffStorageType, 5, 5, 5> *>(nullptr)});

  if (!found) {
    std::string shape_str;
    for (size_t i = 0; i < spec.shape.size(); ++i) {
      if (i > 0)
        shape_str += "x";
      shape_str += std::to_string(spec.shape[i]);
    }
    throw std::invalid_argument("Unsupported stencil shape: " + shape_str +
                                ". Supported shapes: 3, 5, 7, 3x3, 5x5, 7x7, 3x3x3, 5x5x5.");
  }

  return result;
}

CoeffStorageType find_integer_basis(const StencilSpec &spec) {
  CoeffStorageType abs_min = CoeffStorageType(0);
  for (const auto &c : spec.coeffs) {
    CoeffStorageType ac = (c >= CoeffStorageType(0)) ? c : -c;
    if (ac > CoeffStorageType(0) && (abs_min == CoeffStorageType(0) || ac < abs_min))
      abs_min = ac;
  }

  if (abs_min == CoeffStorageType(0))
    return CoeffStorageType(1);

  for (int32_t k = 1; k <= MAX_STENCIL_SLICE_INT; ++k) {
    CoeffStorageType a = abs_min / CoeffStorageType(k);
    bool all_integer = true;
    for (const auto &c : spec.coeffs) {
      CoeffStorageType ratio = c / a;
      CoeffStorageType rounded = (ratio >= CoeffStorageType(0))
          ? CoeffStorageType(static_cast<int64_t>(ratio + CoeffStorageType(0.5)))
          : CoeffStorageType(static_cast<int64_t>(ratio - CoeffStorageType(0.5)));
      CoeffStorageType diff = ratio - rounded;
      if (diff < CoeffStorageType(0)) diff = -diff;
      if (diff > CoeffStorageType(1e-9)) {
        all_integer = false;
        break;
      }
    }
    if (all_integer)
      return a;
  }

  return CoeffStorageType(0);
}

int32_t compute_integer_p_max(const StencilSpec &spec, CoeffStorageType a) {
  int32_t p_max = -1;
  for (const auto &c : spec.coeffs) {
    CoeffStorageType abs_ratio = detail::coeff_abs(detail::coeff_round(c / a));
    if (abs_ratio > CoeffStorageType(0)) {
      int32_t bit_pos = detail::coeff_ilogb(abs_ratio);
      if (bit_pos > p_max)
        p_max = bit_pos;
    }
  }
  return p_max;
}

int64_t compute_msb_l1(const StencilSpec &spec, CoeffStorageType a,
                       int32_t p_max, int32_t extra_msb_shift) {
  constexpr int32_t target_bits = static_cast<int32_t>(CHAR_BIT * sizeof(int8_t)) - 1;
  int32_t shift = std::max(0, p_max + extra_msb_shift - (target_bits - 1));
  int64_t l1 = 0;
  for (const auto &c : spec.coeffs) {
    CoeffStorageType abs_ratio = detail::coeff_abs(detail::coeff_round(c / a));
    l1 += static_cast<int64_t>(detail::coeff_scalbn(abs_ratio, -shift));
  }
  return l1;
}

int compute_m_needed(const StencilSpec &spec, CoeffStorageType a) {
  constexpr int32_t target_bits = static_cast<int32_t>(CHAR_BIT * sizeof(int8_t)) - 1;
  int32_t p_max = compute_integer_p_max(spec, a);
  if (p_max < 0)
    return 1;
  return p_max / target_bits + 1;
}

StencilParams compute_stencil_params(const StencilSpec &spec) {
  StencilParams p;
  constexpr int32_t target_bits = static_cast<int32_t>(CHAR_BIT * sizeof(int8_t)) - 1;

  p.a_coeff = find_integer_basis(spec);
  p.exact_basis = (p.a_coeff != CoeffStorageType(0));

  int32_t coeff_spread = 0;

  if (!p.exact_basis) {
    CoeffStorageType abs_max = CoeffStorageType(0);
    int32_t exp_max_coeff = INT32_MIN;
    int32_t exp_min_coeff = INT32_MAX;
    for (const auto &c : spec.coeffs) {
      CoeffStorageType ac = detail::coeff_abs(c);
      if (ac > abs_max) abs_max = ac;
      if (ac > CoeffStorageType(0)) {
        int32_t e = detail::coeff_ilogb(ac);
        if (e > exp_max_coeff) exp_max_coeff = e;
        if (e < exp_min_coeff) exp_min_coeff = e;
      }
    }
    if (exp_max_coeff >= exp_min_coeff)
      coeff_spread = exp_max_coeff - exp_min_coeff;

    if (abs_max > CoeffStorageType(0)) {
      int exp;
      int32_t mbits = precision_mantissa_bits(detect_coefficient_precision(spec));
#ifdef OZAKI_HAVE_FLOAT128
      frexpq(abs_max, &exp);
      p.a_coeff = scalbnq(static_cast<__float128>(1.0), exp - mbits);
#else
      std::frexp(static_cast<long double>(abs_max), &exp);
      p.a_coeff = std::scalbn(static_cast<long double>(1.0), exp - mbits);
#endif
    } else {
      p.a_coeff = CoeffStorageType(1);
    }
  }

  int32_t p_max = compute_integer_p_max(spec, p.a_coeff);

  p.extra_msb_shift = 0;
  if (p_max >= 0) {
    auto scaling_factor_for = [&](int32_t extra_shift) -> CoeffStorageType {
      int32_t shift0 = std::max(0, p_max - (target_bits - 1) + extra_shift);
      CoeffStorageType eff_a = detail::coeff_scalbn(p.a_coeff, shift0);
      return (eff_a > CoeffStorageType(0)) ? (CoeffStorageType(1) / eff_a) : CoeffStorageType(1);
    };

    while (compute_msb_l1(spec, p.a_coeff, p_max, p.extra_msb_shift) >= 128 ||
           scaling_factor_for(p.extra_msb_shift) >= CoeffStorageType(128)) {
      p.extra_msb_shift++;
    }
  }

  p.m = (p_max < 0) ? 1 : (p_max + p.extra_msb_shift + coeff_spread) / target_bits + 1;

  {
    int32_t shift0 = std::max(0, p_max - (target_bits - 1) + p.extra_msb_shift);
    CoeffStorageType effective_a = detail::coeff_scalbn(p.a_coeff, shift0);
    p.scaling_factor = (effective_a > CoeffStorageType(0))
        ? (CoeffStorageType(1) / effective_a)
        : CoeffStorageType(1);
  }

  return p;
}

const std::vector<StencilSpec> &default_stencils() {
  static const std::vector<StencilSpec> stencils = {
      {"Heat",
       {3, 3},
       {1, 1},
       {CoeffStorageType(0), CoeffStorageType(3.0/16), CoeffStorageType(0), CoeffStorageType(3.0/16), CoeffStorageType(4.0/16),
        CoeffStorageType(3.0/16), CoeffStorageType(0), CoeffStorageType(3.0/16), CoeffStorageType(0)},
       OriginalPrecision::Double},
      {"AdvectionDiffusion",
       {3, 3},
       {1, 1},
       {CoeffStorageType(0), CoeffStorageType(11.0/61), CoeffStorageType(0), CoeffStorageType(11.0/61), CoeffStorageType(15.0/61),
        CoeffStorageType(12.0/61), CoeffStorageType(0), CoeffStorageType(12.0/61), CoeffStorageType(0)},
       OriginalPrecision::Double},
      {"BoxAvg",
       {3, 3},
       {1, 1},
       {CoeffStorageType(1.0/9), CoeffStorageType(1.0/9), CoeffStorageType(1.0/9), CoeffStorageType(1.0/9), CoeffStorageType(1.0/9),
        CoeffStorageType(1.0/9), CoeffStorageType(1.0/9), CoeffStorageType(1.0/9), CoeffStorageType(1.0/9)},
       OriginalPrecision::Double},
      {"BoxRandom",
       {3, 3},
       {1, 1},
       {CoeffStorageType(-3.0/34), CoeffStorageType(2.0/34), CoeffStorageType(1.0/34), CoeffStorageType(-8.0/34), CoeffStorageType(4.0/34),
        CoeffStorageType(-5.0/34), CoeffStorageType(-1.0/34), CoeffStorageType(-1.0/34), CoeffStorageType(9.0/34)},
       OriginalPrecision::Double},
  };
  return stencils;
}

} // namespace ozi
