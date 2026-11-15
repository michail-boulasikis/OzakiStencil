#ifndef OZAKI_GPU_SLICED_TENSOR_CUH
#define OZAKI_GPU_SLICED_TENSOR_CUH

#include "digit_math.cuh"
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <ozaki/core/device_memory.cuh>
#include <ozaki/core/traits.hpp>
#include <ozaki/gpu/gpu_checks.cuh>
#include <span>
#include <stdexcept>
#include <string>

namespace ozaki::gpu {

constexpr int32_t MAX_SLICES = 32;
constexpr int32_t SLICED_TENSOR_ROW_ALIGNMENT = 32;

// Kernel argument for a packed stencil (see packed_stencil.cuh). The
// coefficient slices span significances [rL, rH].
struct PackedStencilArg {
  int32_t rH, rL;
  core::EmulationType *wData;
  const int64_t *positiveSums;
  const int64_t *negativeSums;

  CUDA_HOST CUDA_DEVICE CUDA_INLINE constexpr int32_t active_slices() const { return rH - rL + 1; }
};

struct SliceDescriptor {
  int32_t rH = 0;
  int32_t rL = 1;
  uint32_t sMask = 0U;
  int32_t mssMin = std::numeric_limits<core::EmulationType>::max();
  int32_t mssMax = std::numeric_limits<core::EmulationType>::min();
  int32_t overflowSlices = 0;
};

// Initializes a stencil output descriptor for the selected arithmetic window.
// The halo is homogeneous (zero), so the window is determined by the interior
// arithmetic alone. The extrema start empty and are reduced by the compute
// kernel. Returns false when the window leaves the int32_t range.
CUDA_DEVICE CUDA_INLINE bool seed_output_descriptor(SliceDescriptor *output, int32_t data_slices,
                                                    int64_t arithmetic_rh) {
  const int64_t selected_rl = arithmetic_rh - data_slices + 1;
  if (arithmetic_rh < std::numeric_limits<int32_t>::min() || arithmetic_rh > std::numeric_limits<int32_t>::max() ||
      selected_rl < std::numeric_limits<int32_t>::min() || selected_rl > std::numeric_limits<int32_t>::max()) {
    return false;
  }
  output->rH = static_cast<int32_t>(arithmetic_rh);
  output->rL = static_cast<int32_t>(selected_rl);
  output->sMask = 0U;
  output->mssMin = std::numeric_limits<core::EmulationType>::max();
  output->mssMax = std::numeric_limits<core::EmulationType>::min();
  return true;
}

struct SlicedTensorView {
  int32_t nx = 0, ny = 0, nz = 0;
  size_t rowStrideBytes = 0;
  size_t planeStrideBytes = 0;
  size_t sliceStrideBytes = 0;
  core::EmulationType *sData = nullptr;
  SliceDescriptor *info = nullptr;

  CUDA_HOST CUDA_DEVICE CUDA_INLINE core::EmulationType *ptr(int32_t plane, int32_t x, int32_t y, int32_t z = 0) const {
    return reinterpret_cast<core::EmulationType *>(
        reinterpret_cast<unsigned char *>(sData) + static_cast<size_t>(plane) * sliceStrideBytes +
        static_cast<size_t>(z) * planeStrideBytes + static_cast<size_t>(y) * rowStrideBytes + static_cast<size_t>(x));
  }

  CUDA_HOST CUDA_DEVICE CUDA_INLINE core::EmulationType *ptr_from_linear(int32_t plane, size_t logical) const {
    const size_t xy = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const int32_t z = static_cast<int32_t>(logical / xy);
    const size_t in_plane = logical - static_cast<size_t>(z) * xy;
    const int32_t y = static_cast<int32_t>(in_plane / static_cast<size_t>(nx));
    const int32_t x = static_cast<int32_t>(in_plane - static_cast<size_t>(y) * static_cast<size_t>(nx));
    return ptr(plane, x, y, z);
  }

  CUDA_DEVICE CUDA_INLINE void load_info(SliceDescriptor &dst) const {
#ifdef __CUDACC__
    static_assert(sizeof(SliceDescriptor) % 4 == 0);
    if (threadIdx.x < sizeof(SliceDescriptor) / 4) {
      reinterpret_cast<uint32_t *>(&dst)[threadIdx.x] = reinterpret_cast<const uint32_t *>(info)[threadIdx.x];
    }
#else
    throw std::runtime_error("ERROR: `load_info` called in host code.");
#endif
  }
};

// Input operand of the stencil kernels: device-resident TMA descriptors of
// the tensor (see encode_input_tensor_maps) and its slice descriptor.
struct SlicedTensorTmaView {
  const CUtensorMap *maps = nullptr;
  SliceDescriptor *info = nullptr;
};

class SlicedTensor {
public:
  SlicedTensor() = default;
  SlicedTensor(int32_t nx, int32_t ny, int32_t nz, int32_t data_slices);

  SlicedTensor(const SlicedTensor &) = delete;
  SlicedTensor &operator=(const SlicedTensor &) = delete;
  SlicedTensor(SlicedTensor &&) noexcept = default;
  SlicedTensor &operator=(SlicedTensor &&) noexcept = default;

  [[nodiscard]] SlicedTensorView view() {
    return {nx_, ny_, nz_, row_pitch_bytes_, plane_stride_bytes_, slice_stride_bytes_, data_.get(), info_.get()};
  }

  [[nodiscard]] SlicedTensorTmaView tma_view() {
    return {maps_.get(), info_.get()};
  }

  void set_tensor_maps(std::span<const CUtensorMap> maps) { maps_ = core::DeviceArray<CUtensorMap>(maps); }
  void zero(cudaStream_t stream = nullptr);

  [[nodiscard]] int32_t nx() const { return nx_; }
  [[nodiscard]] int32_t ny() const { return ny_; }
  [[nodiscard]] int32_t nz() const { return nz_; }
  [[nodiscard]] int32_t data_slices() const { return data_slices_; }
  [[nodiscard]] int32_t numerical_slices() const { return data_slices_ + 1; }
  [[nodiscard]] size_t row_pitch_bytes() const { return row_pitch_bytes_; }
  [[nodiscard]] size_t plane_stride_bytes() const { return plane_stride_bytes_; }
  [[nodiscard]] size_t slice_stride_bytes() const { return slice_stride_bytes_; }
  [[nodiscard]] size_t storage_bytes() const { return data_.size(); }
  [[nodiscard]] core::EmulationType *data() { return data_.get(); }
  [[nodiscard]] const core::EmulationType *data() const { return data_.get(); }
  [[nodiscard]] SliceDescriptor *descriptor() { return info_.get(); }
  [[nodiscard]] const SliceDescriptor *descriptor() const { return info_.get(); }
  [[nodiscard]] SliceDescriptor descriptor_host() const { return info_.copy_to_host(); }
  void set_descriptor(const SliceDescriptor &descriptor) { info_.copy_from_host(descriptor); }

private:
  int32_t nx_ = 0, ny_ = 0, nz_ = 0, data_slices_ = 0;
  size_t row_pitch_bytes_ = 0, plane_stride_bytes_ = 0, slice_stride_bytes_ = 0;
  core::DeviceObject<SliceDescriptor> info_{};
  core::DeviceArray<core::EmulationType> data_{};
  core::DeviceArray<CUtensorMap> maps_{};
};

} // namespace ozaki::gpu

#endif // OZAKI_GPU_SLICED_TENSOR_CUH
