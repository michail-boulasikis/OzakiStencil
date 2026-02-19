#ifndef HEADER_BENCHMARK_UTILS_CUH
#define HEADER_BENCHMARK_UTILS_CUH

#include "gpu_checks.cuh"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace ozigpu {

struct L2CacheInvalidator {
  int32_t *d_buffer = nullptr;
  size_t buffer_size = 0;

  L2CacheInvalidator() {
    int device_id;
    int l2_cache_size;
    CUDA_CHECK(cudaGetDevice(&device_id));
    CUDA_CHECK(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, device_id));
    buffer_size = static_cast<size_t>(l2_cache_size) * 2;
    CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));
  }

  ~L2CacheInvalidator() { cudaFree(d_buffer); }

  L2CacheInvalidator(const L2CacheInvalidator &) = delete;
  L2CacheInvalidator &operator=(const L2CacheInvalidator &) = delete;

  void invalidate() {
    NVTX_PUSH("Flushing L2")
    CUDA_CHECK(cudaMemset(d_buffer, 0, buffer_size));
    CUDA_CHECK(cudaDeviceSynchronize());
    NVTX_POP();
  }
};

} // namespace ozigpu

#endif // HEADER_BENCHMARK_UTILS_CUH
