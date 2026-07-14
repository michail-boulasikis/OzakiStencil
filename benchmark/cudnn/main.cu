#include <cuda_runtime.h>
#include <cudnn.h>
#include <ozaki/core/device_memory.cuh>
#include <ozaki/gpu/gpu_checks.cuh>
#include <string>
#include <vector>
#include <iostream>

int main(int argc, char *argv[]) {

  if (argc != 3) {
    std::cerr << "Usage: <cudnn-bench> STENCIL_RADIUS[int] INPUT_SIZE[int]\n";
    return -1;
  }
  int radius = std::stoi(argv[1]);
  int input_size = std::stoi(argv[2]);

  const int N = 1, C = 1, H = input_size, W = input_size;
  const int K = 1, R = 2*radius + 1, S = 2*radius + 1;
  const int pad = radius, stride = 1;
  const int iterations = 128;

  cudnnHandle_t cudnn;
  CUDNN_CHECK(cudnnCreate(&cudnn));

  cudnnTensorDescriptor_t in_desc, out_desc;
  cudnnFilterDescriptor_t filt_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, C, H, W));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, R, S));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad, pad, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION,
                                              CUDNN_DATA_DOUBLE));

  int n_out, c_out, h_out, w_out;
  CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &n_out, &c_out, &h_out, &w_out));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, n_out, c_out, h_out, w_out));

  size_t input_bytes = N * C * H * W * sizeof(double);
  size_t output_bytes = N * c_out * h_out * w_out * sizeof(double);
  size_t filter_bytes = K * C * R * S * sizeof(double);

  core::DeviceArray<double> d_input(input_bytes / sizeof(double));
  core::DeviceArray<double> d_output(output_bytes / sizeof(double));
  core::DeviceArray<double> d_filter(filter_bytes / sizeof(double));

  std::vector<double> h_input(H * W, 1.0);
  std::vector<double> h_filter(R * S, 0.255); // just a random number
  d_input.copy_from_host(h_input.data(), h_input.size());
  d_filter.copy_from_host(h_filter.data(), h_filter.size());

  cudnnConvolutionFwdAlgoPerf_t perf;
  int returnedAlgoCount = 0;

  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, in_desc, filt_desc, conv_desc, out_desc, 1,
                                                     &returnedAlgoCount, &perf));
  if (returnedAlgoCount == 0) {
    std::cerr << "No suitable cuDNN algorithm found!" << std::endl;
    return -1;
  }

  cudnnConvolutionFwdAlgo_t algo = perf.algo;

  // Workspace allocation
  size_t ws_size = 0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

  core::DeviceArray<std::byte> d_workspace;
  if (ws_size > 0)
    d_workspace.reset(ws_size);

  const double alpha = 1.0f, beta = 0.0f;

  // warmup
  for (int i = 0; i < 10; ++i) {
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, in_desc, d_input.get(), filt_desc, d_filter.get(), conv_desc,
                                        algo, d_workspace.get(), ws_size, &beta, out_desc, d_output.get()));
    std::swap(d_input, d_output);
  }

  NVTX_PUSH("INTEGRATION");
  for (int i = 0; i < iterations; ++i) {
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha, in_desc, d_input.get(), filt_desc, d_filter.get(), conv_desc,
                                        algo, d_workspace.get(), ws_size, &beta, out_desc, d_output.get()));
    std::swap(d_input, d_output);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  NVTX_POP();

  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyFilterDescriptor(filt_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(cudnn);

  return 0;
}
