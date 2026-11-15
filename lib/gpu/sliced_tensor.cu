#include <ozaki/gpu/sliced_tensor.cuh>

#include <limits>

namespace ozaki::gpu {
namespace {

size_t checked_mul(size_t a, size_t b, const char *message) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
    throw std::overflow_error(message);
  }
  return a * b;
}

} // namespace

SlicedTensor::SlicedTensor(int32_t nx, int32_t ny, int32_t nz, int32_t data_slices)
    : nx_(nx), ny_(ny), nz_(nz), data_slices_(data_slices) {
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    throw std::invalid_argument("SlicedTensor extents must be positive.");
  }
  if (data_slices < 1 || data_slices > MAX_SLICES - 1) {
    throw std::invalid_argument("SlicedTensor data slice count must be in [1, 31].");
  }
  const size_t logical_row_bytes = static_cast<size_t>(nx) * sizeof(core::EmulationType);
  if (logical_row_bytes > std::numeric_limits<size_t>::max() - static_cast<size_t>(SLICED_TENSOR_ROW_ALIGNMENT - 1)) {
    throw std::overflow_error("SlicedTensor row pitch overflows size_t.");
  }
  row_pitch_bytes_ = align_up(logical_row_bytes, static_cast<size_t>(SLICED_TENSOR_ROW_ALIGNMENT));
  plane_stride_bytes_ =
      checked_mul(row_pitch_bytes_, static_cast<size_t>(ny), "SlicedTensor plane stride overflows size_t.");
  slice_stride_bytes_ =
      checked_mul(plane_stride_bytes_, static_cast<size_t>(nz), "SlicedTensor slice stride overflows size_t.");
  const size_t storage = checked_mul(slice_stride_bytes_, static_cast<size_t>(data_slices + 1),
                                     "SlicedTensor allocation size overflows size_t.");
  data_.reset(storage);
  set_descriptor(SliceDescriptor{});
  zero();
}

void SlicedTensor::zero(cudaStream_t stream) {
  if (data_.empty()) {
    return;
  }
  CUDA_CHECK(cudaMemsetAsync(data_.get(), 0, data_.size_bytes(), stream));
}

} // namespace ozaki::gpu
