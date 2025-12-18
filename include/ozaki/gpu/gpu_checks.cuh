#ifndef HEADER_GPU_CHECKS_CUH
#define HEADER_GPU_CHECKS_CUH
// ============ ONLY CUDA COMPILER SEES THESE DECLARATIONS AND MACROS =============
#ifdef __CUDACC__

#define WARP_SIZE 32
#define CUDA_GLOBAL __global__
#define CUDA_DEVICE __device__
#define CUDA_HOST __host__
#define CUDA_FORCEINLINE __forceinline__

// CUDA/CUBLAS Check macros, only run when building in debug mode.
#if !defined(NDEBUG)
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = (call);                                                                                          \
    if (cudaSuccess != err) {                                                                                          \
      std::cerr << "[ERROR] CUDA: " << cudaGetErrorString(err) << " (code " << err << ") at file " << __FILE__         \
                << " line " << __LINE__ << '\n';                                                                       \
    }                                                                                                                  \
  } while (0);
#define CUBLAS_CHECK(call)                                                                                             \
  do {                                                                                                                 \
    cublasStatus_t err = (call);                                                                                       \
    if (err != CUBLAS_STATUS_SUCCESS) {                                                                                \
      std::cerr << "[ERROR] CUBLAS: " << cublasGetStatusString(err) << " (code " << err << ") at file " << __FILE__    \
                << " line " << __LINE__ << '\n';                                                                       \
    }                                                                                                                  \
  } while (0);
#define CUDNN_CHECK(call)                                                                                              \
  do {                                                                                                                 \
    cudnnStatus_t err = (call);                                                                                        \
    if (err != CUDNN_STATUS_SUCCESS) {                                                                                 \
      std::cerr << "[ERROR] CUDNN: " << cudnnGetErrorString(err) << " (code " << err << ") at file " << __FILE__       \
                << " line " << __LINE__ << '\n';                                                                       \
    }                                                                                                                  \
  } while (0);
#else
#define CUDA_CHECK(call) (call)
#define CUBLAS_CHECK(call) (call)
#define CUDNN_CHECK(call) (call)
#endif // !defined(NDEBUG)

// Option which decides whether to capture NVTX events, helps gather profiling information.
#ifdef OZAKI_NVTX_EVENTS
#include <nvtx3/nvToolsExt.h>
#define NVTX_PUSH(name)                                                                                                \
  do {                                                                                                                 \
    nvtxRangePush(name);                                                                                               \
  } while (0);
#define NVTX_POP()                                                                                                     \
  do {                                                                                                                 \
    nvtxRangePop();                                                                                                    \
  } while (0);
#else
#define NVTX_PUSH(name)
#define NVTX_POP()
#endif // OZAKI_NVTX_EVENTS

#else // __CUDACC__
#define CUDA_GLOBAL
#define CUDA_DEVICE
#define CUDA_HOST
#define CUDA_FORCEINLINE
#endif // __CUDACC__

#endif // HEADER_GPU_CHECKS_CUH
