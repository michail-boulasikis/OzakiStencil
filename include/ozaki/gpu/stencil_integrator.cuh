#ifndef OZAKI_GPU_STENCIL_INTEGRATOR_CUH
#define OZAKI_GPU_STENCIL_INTEGRATOR_CUH

#include "packed_stencil.cuh"

#include <ozaki/core/device_memory.cuh>
#include <ozaki/utils/tensor.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace ozaki::gpu {

#if defined(OZAKI_CUDA_ARCH) && OZAKI_CUDA_ARCH >= 100
using IntegrationReal = __float128;
#else
using IntegrationReal = double;
#endif

class StencilIntegrator {
public:
  StencilIntegrator(const StencilSpec &spec, const Tensor<IntegrationReal> &input, int32_t data_slices);
  ~StencilIntegrator();

  StencilIntegrator(const StencilIntegrator &) = delete;
  StencilIntegrator &operator=(const StencilIntegrator &) = delete;
  StencilIntegrator(StencilIntegrator &&) = delete;
  StencilIntegrator &operator=(StencilIntegrator &&) = delete;

  void reset();
  void integrate(int32_t timesteps);
  void synchronize() const;
  [[nodiscard]] std::vector<IntegrationReal> reconstruct();

  [[nodiscard]] cudaStream_t stream() const { return stream_; }

private:
  void build_tensor_maps();
  void launch_one();

  PackedStencil stencil_;
  Tensor<IntegrationReal> host_input_;
  int32_t data_slices_ = 0;
  size_t logical_elements_ = 0;
  SlicedTensor tensors_[2];
  int32_t current_ = 0;
  int32_t sm_count_ = 1;
  core::DeviceArray<IntegrationReal> dense_input_;
  core::DeviceArray<IntegrationReal> reconstructed_;
  core::DeviceObject<int32_t> scale_;
  cudaStream_t stream_ = nullptr;
};

} // namespace ozaki::gpu

#endif
