#include "weight_layout.hpp"

#include <ozaki/gpu/packed_stencil.cuh>

#include <quadmath.h>

#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ozaki::gpu {
namespace {

constexpr int32_t MAX_COEFFICIENT_SLICES = 9;

// Coefficient digits of one stencil: [slice][coefficient], coefficients in
// row-major ([z][y][x]) order, slice 0 most significant.
using DigitSlices = std::vector<std::vector<core::EmulationType>>;

// Rounds a coefficient to the precision the stencil file specifies.
CoeffStorageType round_to_specified_precision(CoeffStorageType value, OriginalPrecision precision) {
  switch (precision) {
  case OriginalPrecision::Int8:
    return static_cast<CoeffStorageType>(static_cast<int8_t>(value));
  case OriginalPrecision::Float:
    return static_cast<CoeffStorageType>(static_cast<float>(value));
  case OriginalPrecision::Double:
    return static_cast<CoeffStorageType>(static_cast<double>(value));
  case OriginalPrecision::LongDouble:
    return static_cast<CoeffStorageType>(static_cast<long double>(value));
  case OriginalPrecision::Float128:
    break;
  }
  return value;
}

// Splits the coefficients exactly into base-128 int8 digits. Throws if more than nine digits
// are needed.
DigitSlices slice_coefficients(const std::vector<CoeffStorageType> &coefficients, int32_t &rH) {
  using Traits = core::RealTraits<CoeffStorageType>;
  CoeffStorageType absMax = 0;
  for (const CoeffStorageType c : coefficients) {
    if (!Traits::isfinite(c)) {
      throw std::invalid_argument("Stencil coefficients must be finite.");
    }
    absMax = Traits::max(absMax, Traits::abs(c));
  }
  int maxExponent = 0;
  if (absMax != 0) {
    Traits::frexp(absMax, &maxExponent);
  }
  rH = cdiv(maxExponent, core::SLICE_STEP);

  std::vector<CoeffStorageType> residuals(coefficients.size());
  for (size_t i = 0; i < coefficients.size(); ++i) {
    residuals[i] = Traits::scalbn(coefficients[i], -core::SLICE_STEP * rH);
  }
  DigitSlices digits;
  bool exact = false;
  while (!exact && static_cast<int32_t>(digits.size()) < MAX_COEFFICIENT_SLICES) {
    const int32_t scale = core::SLICE_STEP * static_cast<int32_t>(digits.size() + 1);
    std::vector<core::EmulationType> &slice = digits.emplace_back(coefficients.size());
    exact = true;
    for (size_t i = 0; i < coefficients.size(); ++i) {
      slice[i] = Traits::down_cast(residuals[i], scale);
      residuals[i] -= Traits::up_cast(slice[i], scale);
      exact = exact && residuals[i] == 0;
    }
  }
  // A residual can also underflow to zero when the dynamic range is too wide,
  // so verify that the digits reproduce every coefficient.
  for (size_t i = 0; i < coefficients.size(); ++i) {
    CoeffStorageType value = 0;
    for (size_t s = 0; s < digits.size(); ++s) {
      value += Traits::up_cast(digits[s][i], core::SLICE_STEP * static_cast<int32_t>(s + 1));
    }
    if (!exact || Traits::scalbn(value, core::SLICE_STEP * rH) != coefficients[i]) {
      throw std::range_error("Stencil coefficients need more than nine int8 slices.");
    }
  }
  return digits;
}

// Packs 2D slices of side 3, 7 or 15 into Mma2dWeightLayout.
std::vector<core::EmulationType> pack_2d(int32_t side, const DigitSlices &slices) {
  const Mma2dWeightLayout layout{side, static_cast<int32_t>(slices.size())};
  const int32_t P = layout.group_cols();
  std::vector<core::EmulationType> packed(static_cast<size_t>(layout.bytes()), 0);
  for (int32_t slice = 0; slice < layout.slices; ++slice) {
    for (int32_t r = 0; r < side; ++r) {
      for (int32_t c = 0; c < side; ++c) {
        for (int32_t output = 0; output < P; ++output) {
          const int32_t input = output + c;
          packed[static_cast<size_t>(layout.byte_index(input / P, r * P + input % P, slice * P + output))] =
              slices[static_cast<size_t>(slice)][static_cast<size_t>(r * side + c)];
        }
      }
    }
  }
  return packed;
}

std::vector<core::EmulationType> pack_3d(int32_t side, const DigitSlices &slices) {
  const Mma3dWeightLayout layout{static_cast<int32_t>(slices.size())};
  std::vector<core::EmulationType> packed(static_cast<size_t>(layout.bytes()), 0);
  for (int32_t slice = 0; slice < layout.slices; ++slice) {
    for (int32_t z = 0; z < side; ++z) {
      for (int32_t y = 0; y < side; ++y) {
        for (int32_t x = 0; x < side; ++x) {
          for (int32_t output = 0; output < Mma3dWeightLayout::OUTPUTS; ++output) {
            const auto k = Mma3dWeightLayout::k_coordinate(y, output + x);
            packed[static_cast<size_t>(layout.byte_index(z, k.tess, slice, k.k, output))] =
                slices[static_cast<size_t>(slice)][static_cast<size_t>((z * side + y) * side + x)];
          }
        }
      }
    }
  }
  return packed;
}

} // namespace

PackedStencil pack_stencil(const StencilSpec &spec) {
  const int32_t dims = static_cast<int32_t>(spec.shape.size());
  const int32_t side = dims > 0 ? spec.shape.front() : 0;
  size_t count = 1;
  for (int32_t d = 0; d < dims; ++d) {
    if (spec.shape[d] != side || spec.offsets.size() != spec.shape.size() || spec.offsets[d] != side / 2) {
      throw std::invalid_argument("Ozaki stencils must be centered and square/cubic.");
    }
    count *= static_cast<size_t>(side);
  }
  if (!((dims == 2 && (side == 3 || side == 7 || side == 15)) || (dims == 3 && (side == 3 || side == 5)))) {
    throw std::invalid_argument("Unsupported Ozaki stencil shape (supported: 3x3, 7x7, 15x15, 3x3x3, 5x5x5).");
  }
  if (spec.coeffs.size() != count) {
    throw std::invalid_argument("Stencil coefficient count does not match its shape.");
  }

  std::vector<CoeffStorageType> coefficients(spec.coeffs.size());
  for (size_t i = 0; i < coefficients.size(); ++i) {
    coefficients[i] = round_to_specified_precision(spec.coeffs[i], spec.specified_precision);
  }
  int32_t rH = 0;
  const DigitSlices digits = slice_coefficients(coefficients, rH);
  const int32_t slices = static_cast<int32_t>(digits.size());
  if (slices != 1 && slices != 2 && slices != 9 && !(slices == 3 && dims == 2 && side == 15)) {
    throw std::invalid_argument("Stencil '" + spec.name + "' has " + std::to_string(slices) +
                                " coefficient slices; supported are 1, 2 and 9 (and 3 for 15x15).");
  }

  // Per slice (least significant first): sum of positive, then of negative digits.
  std::vector<int64_t> sums(2 * static_cast<size_t>(slices), 0);
  for (int32_t weight = 0; weight < slices; ++weight) {
    for (const core::EmulationType digit : digits[static_cast<size_t>(slices - 1 - weight)]) {
      sums[static_cast<size_t>(digit >= 0 ? weight : slices + weight)] += digit;
    }
  }

  const std::vector<core::EmulationType> packed = dims == 2 ? pack_2d(side, digits) : pack_3d(side, digits);
  PackedStencil result;
  result.dims = dims;
  result.side = side;
  result.slices = slices;
  result.rH = rH;
  result.rL = rH - slices + 1;
  result.weights = core::DeviceArray<core::EmulationType>{std::span<const core::EmulationType>(packed)};
  result.intervalSums = core::DeviceArray<int64_t>{std::span<const int64_t>(sums)};
  return result;
}

} // namespace ozaki::gpu
