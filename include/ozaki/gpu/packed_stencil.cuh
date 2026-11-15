#ifndef OZAKI_GPU_PACKED_STENCIL_CUH
#define OZAKI_GPU_PACKED_STENCIL_CUH

#include "sliced_tensor.cuh"

#include <ozaki/core/device_memory.cuh>
#include <ozaki/utils/stencil_spec.hpp>

#include <cstdint>

namespace ozaki::gpu {

// A stencil sliced into `slices` int8 coefficient digits and packed into the
// MMA B-operand layout of its kernel (see weight_layout.hpp).
struct PackedStencil {
  int32_t dims = 0;
  int32_t side = 0;
  int32_t slices = 0; // M: 1, 2, 3 (15x15 only) or 9
  int32_t rH = 0, rL = 1;
  core::DeviceArray<core::EmulationType> weights;
  // Per coefficient slice (least significant first): sum of the positive
  // coefficients, then sum of the negative ones. Used to bound carries.
  core::DeviceArray<int64_t> intervalSums;

  [[nodiscard]] int32_t halo() const { return side / 2; }
  [[nodiscard]] PackedStencilArg arg() {
    return {rH, rL, weights.get(), intervalSums.get(), intervalSums.get() + slices};
  }
};

// Validates, slices and packs a centered square/cubic stencil of a supported
// shape (3x3, 7x7, 15x15, 3x3x3, 5x5x5) with 1, 2 or 9 coefficient slices (also
// 3 for 15x15). Throws std::invalid_argument / std::range_error otherwise.
[[nodiscard]] PackedStencil pack_stencil(const StencilSpec &spec);

} // namespace ozaki::gpu

#endif // OZAKI_GPU_PACKED_STENCIL_CUH
