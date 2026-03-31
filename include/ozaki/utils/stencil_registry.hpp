#ifndef HEADER_STENCIL_REGISTRY_HPP
#define HEADER_STENCIL_REGISTRY_HPP

#include "stencil_spec.hpp"
#include "../core/stencil.hpp"

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace ozi {

static constexpr int32_t MAX_STENCIL_SLICE_INT = (1 << 21);

// Number of significand bits (including implicit leading 1) for each
// OriginalPrecision level.  Used by the power-of-2 fallback in
// compute_stencil_params to match the slicing depth to the *actual*
// coefficient precision rather than the storage type.
inline int32_t precision_mantissa_bits(OriginalPrecision p) {
  switch (p) {
  case OriginalPrecision::Int8:        return 7;
  case OriginalPrecision::Float:       return 24;
  case OriginalPrecision::Double:      return 53;
  case OriginalPrecision::LongDouble:  return 64;
  case OriginalPrecision::Float128:    return 113;
  }
  return 53; // fallback
}

namespace detail {
inline CoeffStorageType coeff_abs(CoeffStorageType x) {
  return (x >= CoeffStorageType(0)) ? x : -x;
}
inline CoeffStorageType coeff_round(CoeffStorageType x) {
#ifdef OZAKI_HAVE_FLOAT128
  return roundq(x);
#else
  return std::round(x);
#endif
}
inline int32_t coeff_ilogb(CoeffStorageType x) {
#ifdef OZAKI_HAVE_FLOAT128
  return static_cast<int32_t>(ilogbq(x));
#else
  return std::ilogb(x);
#endif
}
inline CoeffStorageType coeff_scalbn(CoeffStorageType x, int32_t n) {
#ifdef OZAKI_HAVE_FLOAT128
  return scalbnq(x, n);
#else
  return std::scalbn(x, n);
#endif
}
} // namespace detail

// Variant over all supported stencil shapes.
// Adding a new shape requires one line here and a recompile.
using StencilVariant = std::variant<
    Stencil<CoeffStorageType, 3>,
    Stencil<CoeffStorageType, 5>,
    Stencil<CoeffStorageType, 7>,
    Stencil<CoeffStorageType, 3, 3>,
    Stencil<CoeffStorageType, 5, 5>,
    Stencil<CoeffStorageType, 7, 7>,
    Stencil<CoeffStorageType, 3, 3, 3>,
    Stencil<CoeffStorageType, 5, 5, 5>
>;

// Creates a typed Stencil from a runtime StencilSpec.
// Throws std::invalid_argument if the shape is not in the registry.
inline StencilVariant make_stencil(const StencilSpec &spec) {
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

  // Try each variant alternative
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

  // Expand over all alternatives
  std::apply([&](auto *...tags) { (try_variant(tags), ...); },
             std::tuple<
                 Stencil<CoeffStorageType, 3> *,
                 Stencil<CoeffStorageType, 5> *,
                 Stencil<CoeffStorageType, 7> *,
                 Stencil<CoeffStorageType, 3, 3> *,
                 Stencil<CoeffStorageType, 5, 5> *,
                 Stencil<CoeffStorageType, 7, 7> *,
                 Stencil<CoeffStorageType, 3, 3, 3> *,
                 Stencil<CoeffStorageType, 5, 5, 5> *
             >{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr});

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

// Finds a_coeff > 0 such that all spec.coeffs[i] / a_coeff are integers.
// Uses brute force: tries a = abs_min / k for k = 1..MAX_STENCIL_SLICE_INT.
// Returns CoeffStorageType(1) if all coefficients are zero.
// Returns CoeffStorageType(0) as a sentinel if no exact integer basis is found.
inline CoeffStorageType find_integer_basis(const StencilSpec &spec) {
  // Find the smallest positive coefficient magnitude
  CoeffStorageType abs_min = CoeffStorageType(0);
  for (const auto &c : spec.coeffs) {
    CoeffStorageType ac = (c >= CoeffStorageType(0)) ? c : -c;
    if (ac > CoeffStorageType(0) && (abs_min == CoeffStorageType(0) || ac < abs_min))
      abs_min = ac;
  }

  // Degenerate case: all coefficients are zero
  if (abs_min == CoeffStorageType(0))
    return CoeffStorageType(1);

  // Brute-force: find smallest k such that abs_min/k evenly divides all coefficients
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
    if (all_integer) {
      return a;
    }
  }

  return CoeffStorageType(0); // sentinel: no exact integer basis found
}

// Returns the 0-indexed MSB position of the largest |round(c/a)| across all coefficients.
// Returns -1 if all coefficients round to zero.
inline int32_t compute_integer_p_max(const StencilSpec &spec, CoeffStorageType a) {
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

// Returns the L1 norm of the would-be MSB stencil slice for a given extra_msb_shift.
inline int64_t compute_msb_l1(const StencilSpec &spec, CoeffStorageType a,
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

inline int compute_m_needed(const StencilSpec &spec, CoeffStorageType a) {
  constexpr int32_t target_bits = static_cast<int32_t>(CHAR_BIT * sizeof(int8_t)) - 1;
  int32_t p_max = compute_integer_p_max(spec, a);
  if (p_max < 0)
    return 1;
  return p_max / target_bits + 1;
}

// Derived parameters needed to run an Ozaki stencil computation.
struct StencilParams {
  CoeffStorageType a_coeff;        // Integer basis: all coeffs / a_coeff are integers (exact or approx)
  int32_t          m;              // Number of int8 slices required to represent the integer stencil
  CoeffStorageType scaling_factor; // 1 / effective_a; used to undo integer scaling after convolution
  bool             exact_basis;    // true if a_coeff is exact; false if the fallback approximation was used
  int32_t          extra_msb_shift = 0; // Right-shift for MSB slice to keep L1 < 128
};

// Computes all derived parameters needed to run an Ozaki stencil computation from a StencilSpec.
inline StencilParams compute_stencil_params(const StencilSpec &spec) {
  StencilParams p;
  constexpr int32_t target_bits = static_cast<int32_t>(CHAR_BIT * sizeof(int8_t)) - 1;

  // Step 1: find a_coeff such that all coefficients / a_coeff are integers.
  p.a_coeff = find_integer_basis(spec);
  p.exact_basis = (p.a_coeff != CoeffStorageType(0));

  if (!p.exact_basis) {
    // Fallback: power-of-2 normalization (Ozaki-style slicing of the stencil).
    CoeffStorageType abs_max = CoeffStorageType(0);
    for (const auto &c : spec.coeffs) {
      CoeffStorageType ac = detail::coeff_abs(c);
      if (ac > abs_max) abs_max = ac;
    }
    if (abs_max > CoeffStorageType(0)) {
      int exp;
      int32_t mbits = precision_mantissa_bits(spec.specified_precision);
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

  // Step 2: compute p_max (MSB position of the largest integer coefficient).
  int32_t p_max = compute_integer_p_max(spec, p.a_coeff);

  // Step 3: determine extra_msb_shift.
  // Two constraints must hold simultaneously:
  //   (a) L1 norm of the MSB stencil slice < 128 (so overflow_danger_threshold >= 0)
  //   (b) scaling_factor < 128 (so boundary_gain's MSB digit fits in int8_t)
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

  // Step 4: compute m (number of INT8 stencil slices).
  p.m = (p_max < 0) ? 1 : (p_max + p.extra_msb_shift) / target_bits + 1;

  // Step 5: compute scaling_factor = 1 / effective_a.
  // effective_a = a_coeff * 2^shift0, where shift0 accounts for the MSB right-shift.
  {
    int32_t shift0 = std::max(0, p_max - (target_bits - 1) + p.extra_msb_shift);
    CoeffStorageType effective_a = detail::coeff_scalbn(p.a_coeff, shift0);
    p.scaling_factor = (effective_a > CoeffStorageType(0))
        ? (CoeffStorageType(1) / effective_a)
        : CoeffStorageType(1);
  }

  return p;
}

inline const std::vector<StencilSpec> &default_stencils() {
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

#endif // HEADER_STENCIL_REGISTRY_HPP
