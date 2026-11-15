#ifndef OZAKI_GPU_GPU_CHECKS_CUH
#define OZAKI_GPU_GPU_CHECKS_CUH

#include <ozaki/core/traits.hpp>

#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#define WARP_SIZE 32

// Aborts with a message when a CUDA runtime call fails.
#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    const cudaError_t err = (call);                                                                                    \
    if (err != cudaSuccess) {                                                                                          \
      std::cerr << "[FATAL] CUDA: " << cudaGetErrorString(err) << " (code " << err << ") at " << __FILE__ << ":"       \
                << __LINE__ << '\n';                                                                                   \
      std::abort();                                                                                                    \
    }                                                                                                                  \
  } while (0)

#endif // OZAKI_GPU_GPU_CHECKS_CUH
