#ifndef HEADER_STENCIL_REGISTRY_HPP
#define HEADER_STENCIL_REGISTRY_HPP

#include "stencil_spec.hpp"
#include "../core/stencil.hpp"

#include <bit>
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

inline int compute_m_needed(const StencilSpec &spec, CoeffStorageType a) {
  constexpr int32_t target_bits = static_cast<int32_t>(CHAR_BIT * sizeof(int8_t)) - 1;

  int32_t range_msb = 64;
  for (const auto &c : spec.coeffs) {
    CoeffStorageType raw = c / a;
    CoeffStorageType shifted = raw + (raw >= CoeffStorageType(0) ? CoeffStorageType(0.5) : CoeffStorageType(-0.5));
    int64_t int_val = static_cast<int64_t>(shifted);
    uint64_t abs_val = (int_val < 0) ? static_cast<uint64_t>(-int_val) : static_cast<uint64_t>(int_val);
    if (int32_t msb = std::countl_zero(abs_val); msb < range_msb)
      range_msb = msb;
  }

  int32_t p_max = 63 - range_msb;
  if (p_max < 0)
    return 1; // All coefficients rounded to zero
  return p_max / target_bits + 1;
}

// Derived parameters needed to run an Ozaki stencil computation.
struct StencilParams {
  CoeffStorageType a_coeff;        // Integer basis: all coeffs / a_coeff are integers (exact or approx)
  int32_t          m;              // Number of int8 slices required to represent the integer stencil
  CoeffStorageType scaling_factor; // 1 / effective_a; used to undo integer scaling after convolution
  bool             exact_basis;    // true if a_coeff is exact; false if the fallback approximation was used
  int32_t          extra_msb_shift = 0; // Right-shift for MSB slice to keep L1 < 127
};

// Computes all derived parameters needed to run an Ozaki stencil computation from a StencilSpec.
// Wraps find_integer_basis (with fallback) + compute_m_needed + L1-norm / scaling_factor in one call.
inline StencilParams compute_stencil_params(const StencilSpec &spec) {
  StencilParams p;

  // Step 1: find a_coeff such that all coefficients / a_coeff are integers.
  p.a_coeff = find_integer_basis(spec);
  p.exact_basis = (p.a_coeff != CoeffStorageType(0));

  if (!p.exact_basis) {
    // Fallback: power-of-2 normalization (Ozaki-style slicing of the stencil).
    CoeffStorageType abs_max = CoeffStorageType(0);
    for (const auto &c : spec.coeffs) {
      CoeffStorageType ac = (c >= CoeffStorageType(0)) ? c : -c;
      if (ac > abs_max) abs_max = ac;
    }
    if (abs_max > CoeffStorageType(0)) {
      int exp;
#ifdef __GNUC__
      frexpq(abs_max, &exp);
      p.a_coeff = scalbnq(static_cast<__float128>(1.0), exp - 63);
#else
      std::frexp(static_cast<long double>(abs_max), &exp);
      p.a_coeff = std::scalbn(static_cast<long double>(1.0), exp - 63);
#endif
    } else {
      p.a_coeff = CoeffStorageType(1);
    }
    fprintf(stderr, "Note: no exact integer basis found for stencil '%s'; "
                    "using Ozaki slicing (M may be higher)\n",
            spec.name.c_str());
  }

  // Step 2: determine the number of int8 slices needed.
  p.m = compute_m_needed(spec, p.a_coeff);

  // Step 2b: check whether the MSB stencil slice L1 norm >= 127.
  // If so, right-shift the MSB slice by extra_msb_shift bits and add
  // compensating LSB slices so that the overflow threshold stays valid.
  constexpr int32_t target_bits = static_cast<int32_t>(CHAR_BIT * sizeof(int8_t)) - 1;
  int32_t p_max = -1;
  {
    // Compute integer coefficients (same logic as slice_stencil)
    int32_t range_msb = 64;
    std::vector<int64_t> int_coeffs(spec.coeffs.size());
    for (size_t i = 0; i < spec.coeffs.size(); ++i) {
      CoeffStorageType raw = spec.coeffs[i] / p.a_coeff;
      CoeffStorageType shifted = raw + (raw >= CoeffStorageType(0) ? CoeffStorageType(0.5) : CoeffStorageType(-0.5));
      int_coeffs[i] = static_cast<int64_t>(shifted);
      uint64_t abs_val = (int_coeffs[i] < 0) ? static_cast<uint64_t>(-int_coeffs[i]) : static_cast<uint64_t>(int_coeffs[i]);
      if (int32_t msb = std::countl_zero(abs_val); msb < range_msb)
        range_msb = msb;
    }
    p_max = 63 - range_msb;

    // Compute L1 norm of the would-be MSB slice
    p.extra_msb_shift = 0;
    if (p_max >= 0) {
      auto compute_msb_l1 = [&](int32_t extra_shift) -> int64_t {
        int32_t shift = std::max(0, p_max + extra_shift - (target_bits - 1));
        int64_t l1 = 0;
        for (size_t i = 0; i < int_coeffs.size(); ++i) {
          uint64_t abs_val = (int_coeffs[i] < 0)
            ? static_cast<uint64_t>(-int_coeffs[i])
            : static_cast<uint64_t>(int_coeffs[i]);
          l1 += static_cast<int64_t>(abs_val >> shift);
        }
        return l1;
      };

      while (compute_msb_l1(p.extra_msb_shift) >= 63) {
        p.extra_msb_shift++;
      }

      if (p.extra_msb_shift > 0) {
        // Recompute M: the extra shift adds more bits to represent, requiring more slices
        p.m = (p_max + p.extra_msb_shift) / target_bits + 1;
      }
    }
  }

  // Step 3: compute the scaling factor.
  // The digit-domain stencil has per-iteration growth = 1 / effective_a,
  // where effective_a = a_coeff * 2^shift0.
  // total_integration_scale_factor_quad(scaling_factor, N, shifts) = 2^(7*shifts) / scaling_factor^N
  CoeffStorageType effective_a = p.a_coeff;
  {
    int32_t shift0 = std::max(0, p_max - (target_bits - 1) + p.extra_msb_shift);
#ifdef __GNUC__
    effective_a = scalbnq(p.a_coeff, shift0);
#else
    effective_a = std::scalbn(p.a_coeff, shift0);
#endif
  }
  p.scaling_factor = (effective_a > CoeffStorageType(0))
      ? (CoeffStorageType(1) / effective_a)
      : CoeffStorageType(1);

  // Ensure boundary_gain's most-significant digit fits in int8_t (values in [-127, 127]).
  // boundary_gain encodes round(1/effective_a) in base-128; the MSB digit equals
  // floor(scaling_factor). When scaling_factor >= 128 that digit overflows int8_t
  // and gets silently truncated (mod 128).
  // Each extra_msb_shift increment doubles effective_a and halves scaling_factor until < 128.
  while (p.scaling_factor >= CoeffStorageType(128)) {
    p.extra_msb_shift++;
    int32_t new_shift0 = std::max(0, p_max - (target_bits - 1) + p.extra_msb_shift);
#ifdef __GNUC__
    effective_a = scalbnq(p.a_coeff, new_shift0);
#else
    effective_a = std::scalbn(p.a_coeff, new_shift0);
#endif
    p.m = (p_max + p.extra_msb_shift) / target_bits + 1;
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
