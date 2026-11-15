#include <ozaki/gpu/stencil_integrator.cuh>

#include "slicing_kernels.cuh"
#include <ozaki/utils/zero_boundary.hpp>

#include "stencil_kernels.cuh"

#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ozaki::gpu {
namespace {

void require_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void require_driver(CUresult status, const char *operation) {
  if (status == CUDA_SUCCESS) {
    return;
  }
  const char *name = nullptr;
  const char *description = nullptr;
  (void)cuGetErrorName(status, &name);
  (void)cuGetErrorString(status, &description);
  throw std::runtime_error(std::string(operation) + ": " + (name != nullptr ? name : "CUDA driver error") +
                           (description != nullptr ? std::string(" (") + description + ")" : std::string{}));
}

size_t checked_volume(const std::vector<size_t> &shape) {
  size_t result = 1;
  for (const size_t extent : shape) {
    if (extent == 0 || result > std::numeric_limits<size_t>::max() / extent) {
      throw std::overflow_error("Tensor volume is invalid or overflows size_t.");
    }
    result *= extent;
  }
  return result;
}

} // namespace

StencilIntegrator::StencilIntegrator(const StencilSpec &spec, const Tensor<IntegrationReal> &input, int32_t data_slices)
    : stencil_(pack_stencil(spec)), data_slices_(data_slices) {
  if (data_slices < 1 || data_slices > MAX_SLICES - 1) {
    throw std::invalid_argument("Requested data slices must be in [1, 31].");
  }
  if (input.shape.size() != static_cast<size_t>(stencil_.dims)) {
    throw std::invalid_argument("Input tensor rank does not match stencil rank.");
  }
  // Homogeneous Dirichlet boundary.
  host_input_ = zero_boundary_copy(input, static_cast<size_t>(stencil_.halo()));
  logical_elements_ = checked_volume(host_input_.shape);
  if (logical_elements_ != host_input_.data.size()) {
    throw std::invalid_argument("Input tensor shape does not match its storage size.");
  }
  for (const IntegrationReal value : host_input_.data) {
    if (!core::RealTraits<IntegrationReal>::isfinite(value)) {
      throw std::invalid_argument("Ozaki slicing requires finite input values.");
    }
  }

  int device = 0;
  require_cuda(cudaGetDevice(&device), "querying the CUDA device");
  int major = 0;
  require_cuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device),
               "querying CUDA compute capability");
  if (major < 9) {
    throw std::runtime_error("OzakiStencil tensor-core kernels require CUDA compute capability 9.0 or newer.");
  }
  require_cuda(cudaDeviceGetAttribute(&sm_count_, cudaDevAttrMultiProcessorCount, device),
               "querying CUDA multiprocessor count");

  const int32_t nx = static_cast<int32_t>(host_input_.shape.back());
  const int32_t ny = static_cast<int32_t>(host_input_.shape[host_input_.shape.size() - 2]);
  const int32_t nz = stencil_.dims == 3 ? static_cast<int32_t>(host_input_.shape.front()) : 1;
  if (static_cast<size_t>(nx) != host_input_.shape.back() ||
      static_cast<size_t>(ny) != host_input_.shape[host_input_.shape.size() - 2] ||
      (stencil_.dims == 3 && static_cast<size_t>(nz) != host_input_.shape.front())) {
    throw std::overflow_error("Tensor extents exceed int32_t kernel limits.");
  }
  if (nx <= 2 * stencil_.halo() || ny <= 2 * stencil_.halo() || (stencil_.dims == 3 && nz <= 2 * stencil_.halo())) {
    throw std::invalid_argument("Stencil integration requires a non-empty interior.");
  }

  tensors_[0] = SlicedTensor(nx, ny, nz, data_slices);
  tensors_[1] = SlicedTensor(nx, ny, nz, data_slices);
  dense_input_.reset(logical_elements_);
  reconstructed_.reset(logical_elements_);

  require_cuda(cudaStreamCreate(&stream_), "creating the integration stream");
  try {
    require_driver(cuInit(0), "initializing the CUDA driver");
    build_tensor_maps();
    reset();
  } catch (...) {
    (void)cudaStreamDestroy(stream_);
    stream_ = nullptr;
    throw;
  }
}

StencilIntegrator::~StencilIntegrator() {
  if (stream_ != nullptr) {
    (void)cudaStreamSynchronize(stream_);
    (void)cudaStreamDestroy(stream_);
  }
}

void StencilIntegrator::build_tensor_maps() {
  for (SlicedTensor &tensor : tensors_) {
    std::vector<CUtensorMap> maps;
    require_driver(stencil_.dims == 2 ? encode_input_tensor_maps_2d(stencil_, tensor, maps)
                                           : encode_input_tensor_maps_3d(stencil_, tensor, maps),
                   "encoding input tensor maps");
    tensor.set_tensor_maps(std::span<const CUtensorMap>(maps));
  }
}

void StencilIntegrator::reset() {
  synchronize();
  current_ = 0;
  tensors_[0].zero(stream_);
  tensors_[1].zero(stream_);
  require_cuda(cudaMemcpyAsync(dense_input_.get(), host_input_.data.data(), dense_input_.size_bytes(),
                               cudaMemcpyHostToDevice, stream_),
               "uploading the input tensor");

  const int32_t scale_identity = std::numeric_limits<int32_t>::min();
  scale_.copy_from_host(scale_identity);
  tensors_[0].set_descriptor(SliceDescriptor{});
  tensors_[1].set_descriptor(SliceDescriptor{});

  const int32_t max_blocks = std::max(1, sm_count_ * 4);
  const int32_t msb_blocks = std::min(
      max_blocks, std::max(1, cdiv(static_cast<int32_t>(std::min(
                                       logical_elements_, static_cast<size_t>(std::numeric_limits<int32_t>::max()))),
                                   SLICING_THREADS)));
  _msb<IntegrationReal>
      <<<msb_blocks, SLICING_THREADS, 0, stream_>>>(scale_.get(), dense_input_.get(), logical_elements_);
  require_cuda(cudaGetLastError(), "launching MSB selection");
  const int32_t slice_blocks = std::min(
      max_blocks, std::max(1, cdiv(static_cast<int32_t>(std::min(
                                       logical_elements_, static_cast<size_t>(std::numeric_limits<int32_t>::max()))),
                                   SLICING_THREADS)));
  _slice<IntegrationReal><<<slice_blocks, SLICING_THREADS, 0, stream_>>>(
      tensors_[0].view(), dense_input_.get(), scale_.get(), data_slices_, logical_elements_);
  require_cuda(cudaGetLastError(), "launching ordered slicing");
  synchronize();
}

void StencilIntegrator::launch_one() {
  const int32_t destination = current_ ^ 1;
  PackedStencil &stencil = stencil_;
  require_cuda(stencil_.dims == 2
                   ? launch_stencil_2d(stencil, tensors_[destination], tensors_[current_], stream_)
                   : launch_stencil_3d(stencil, tensors_[destination], tensors_[current_], stream_),
               "launching the Ozaki stencil step");
  current_ = destination;
}

void StencilIntegrator::integrate(int32_t timesteps) {
  if (timesteps < 0) {
    throw std::invalid_argument("Timesteps cannot be negative.");
  }
  for (int32_t step = 0; step < timesteps; ++step) {
    launch_one();
  }
}

void StencilIntegrator::synchronize() const {
  if (stream_ != nullptr) {
    require_cuda(cudaStreamSynchronize(stream_), "synchronizing the integration stream");
  }
}

std::vector<IntegrationReal> StencilIntegrator::reconstruct() {
  const int32_t blocks =
      cdiv(static_cast<int32_t>(std::min(logical_elements_, static_cast<size_t>(std::numeric_limits<int32_t>::max()))),
           SLICING_THREADS);
  auto source = tensors_[current_].view();
  _reconstruct<IntegrationReal>
      <<<blocks, SLICING_THREADS, 0, stream_>>>(reconstructed_.get(), source, data_slices_, logical_elements_);
  require_cuda(cudaGetLastError(), "launching reconstruction");
  std::vector<IntegrationReal> result(logical_elements_);
  require_cuda(cudaMemcpyAsync(result.data(), reconstructed_.get(), reconstructed_.size_bytes(), cudaMemcpyDeviceToHost,
                               stream_),
               "copying reconstructed tensor");
  synchronize();
  // A kernel preamble that cannot select a valid output window marks the
  // descriptor invalid; every later step then yields zeros. Fail loudly.
  if (tensors_[current_].descriptor_host().overflowSlices < 0) {
    throw std::runtime_error("Ozaki integration failed: no valid output window could be selected "
                             "(the digit window overflowed or left the representable range).");
  }
  return result;
}

} // namespace ozaki::gpu
