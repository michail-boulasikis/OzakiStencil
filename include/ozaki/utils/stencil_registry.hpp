#ifndef HEADER_STENCIL_REGISTRY_HPP
#define HEADER_STENCIL_REGISTRY_HPP

#include "stencil_spec.hpp"
#include "../core/stencil.hpp"

#include <cstdint>
#include <cstddef>
#include <variant>

namespace ozi {

static constexpr int32_t MAX_STENCIL_SLICE_INT = (1 << 21);

int32_t precision_mantissa_bits(OriginalPrecision p);
OriginalPrecision detect_coefficient_precision(const StencilSpec &spec);

namespace detail {
CoeffStorageType coeff_abs(CoeffStorageType x);
CoeffStorageType coeff_round(CoeffStorageType x);
int32_t coeff_ilogb(CoeffStorageType x);
CoeffStorageType coeff_scalbn(CoeffStorageType x, int32_t n);
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
StencilVariant make_stencil(const StencilSpec &spec);

// Finds a_coeff > 0 such that all spec.coeffs[i] / a_coeff are integers.
// Uses brute force: tries a = abs_min / k for k = 1..MAX_STENCIL_SLICE_INT.
// Returns CoeffStorageType(1) if all coefficients are zero.
// Returns CoeffStorageType(0) as a sentinel if no exact integer basis is found.
CoeffStorageType find_integer_basis(const StencilSpec &spec);

// Returns the 0-indexed MSB position of the largest |round(c/a)| across all coefficients.
// Returns -1 if all coefficients round to zero.
int32_t compute_integer_p_max(const StencilSpec &spec, CoeffStorageType a);

// Returns the L1 norm of the would-be MSB stencil slice for a given extra_msb_shift.
int64_t compute_msb_l1(const StencilSpec &spec, CoeffStorageType a,
                       int32_t p_max, int32_t extra_msb_shift);

int compute_m_needed(const StencilSpec &spec, CoeffStorageType a);

// Derived parameters needed to run an Ozaki stencil computation.
struct StencilParams {
  CoeffStorageType a_coeff;        // Integer basis: all coeffs / a_coeff are integers (exact or approx)
  int32_t          m;              // Number of int8 slices required to represent the integer stencil
  CoeffStorageType scaling_factor; // 1 / effective_a; used to undo integer scaling after convolution
  bool             exact_basis;    // true if a_coeff is exact; false if the fallback approximation was used
  int32_t          extra_msb_shift = 0; // Right-shift for MSB slice to keep L1 < 128
};

// Computes all derived parameters needed to run an Ozaki stencil computation from a StencilSpec.
StencilParams compute_stencil_params(const StencilSpec &spec);

} // namespace ozi

#endif // HEADER_STENCIL_REGISTRY_HPP
