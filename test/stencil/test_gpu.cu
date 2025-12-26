#include <gtest/gtest.h>

#include <ozaki/core/stencil.hpp>
#include <ozaki/gpu/gpu_checks.cuh>
#include <ozaki/gpu/kernels.cuh>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

namespace test_data {
static constexpr uint64_t SEED = 0x123456789;
static constexpr uint32_t CVAL = 0xDEADBEEF;
static constexpr size_t N = 1024 * 1024;
static constexpr size_t N_SMALL = 1024;
} // namespace test_data

class OzakiTestGPU : public ::testing::Test {
protected:
  void SetUp() override {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No CUDA-capable device found. Skipping test.";
    }

    cudaSetDevice(0);
  }
};

TEST_F(OzakiTestGPU, Fill) {
  uint32_t *d_fill;
  uint32_t *h_fill = new uint32_t[test_data::N];
  ASSERT_EQ(CUDA_SUCCESS, cudaMalloc(&d_fill, test_data::N * sizeof(uint32_t)));
  ozigpu::fill<uint32_t><<<1024, 1024>>>(d_fill, test_data::CVAL, test_data::N);
  ASSERT_EQ(CUDA_SUCCESS, cudaDeviceSynchronize());
  ASSERT_EQ(CUDA_SUCCESS, cudaMemcpy(h_fill, d_fill, test_data::N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  for (uint32_t i = 0; i < test_data::N; ++i)
    ASSERT_EQ(test_data::CVAL, h_fill[i]) << "Iter[" << i << "]";
}

TEST_F(OzakiTestGPU, ExtractAbsMaxExpNormal) {
  std::mt19937 engine{test_data::SEED};
  std::vector<double> hdata(test_data::N);

  std::normal_distribution<double> normal_dist(0.0, 1.0);

  for (size_t i = 0; i < test_data::N; ++i) {
    hdata[i] = normal_dist(engine);
  }

  double *ddata;
  int32_t hexp, *dexp;
  int32_t hexp_real = std::numeric_limits<int32_t>::min();
  ASSERT_EQ(CUDA_SUCCESS, cudaMalloc(&ddata, test_data::N * sizeof(double)));
  ASSERT_EQ(CUDA_SUCCESS, cudaMalloc(&dexp, sizeof(int32_t)));
  ASSERT_EQ(CUDA_SUCCESS, cudaMemcpy(ddata, hdata.data(), test_data::N * sizeof(double), cudaMemcpyHostToDevice));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  uint32_t nBlocksReduce = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / RDC_THREADS_PER_BLOCK;
  ozigpu::abs_max_nextpow2_reduce<double><<<nBlocksReduce, RDC_THREADS_PER_BLOCK>>>(dexp, ddata, test_data::N);
  for (const auto x : hdata) {
    if (const int32_t e = static_cast<int32_t>(std::ceil(std::log2(std::abs(x)))); e > hexp_real)
      hexp_real = e;
  }
  ASSERT_EQ(CUDA_SUCCESS, cudaDeviceSynchronize());
  ASSERT_EQ(CUDA_SUCCESS, cudaMemcpy(&hexp, dexp, sizeof(int32_t), cudaMemcpyDeviceToHost));
  ASSERT_EQ(hexp_real, hexp);

  cudaFree(ddata);
  cudaFree(dexp);
}

TEST_F(OzakiTestGPU, ExtractAbsMaxExpBeta) {
  std::mt19937 engine{test_data::SEED};
  std::vector<double> hdata(test_data::N);

  std::gamma_distribution<double> gamma_dist(0.2, 1.0);
  std::bernoulli_distribution bool_dist(0.5);

  for (size_t i = 0; i < test_data::N; ++i) {
    const double g1 = gamma_dist(engine);
    const double g2 = gamma_dist(engine);

    const double beta_val = g1 / (g1 + g2);
    const double scalar =
        bool_dist(engine) ? (1 / std::numeric_limits<double>::epsilon()) : std::numeric_limits<double>::epsilon();
    const double scaled_val = beta_val * scalar;

    hdata[i] = bool_dist(engine) ? scaled_val : -scaled_val;
  }

  double *ddata;
  int32_t hexp, *dexp;
  int32_t hexp_real = std::numeric_limits<int32_t>::min();
  ASSERT_EQ(CUDA_SUCCESS, cudaMalloc(&ddata, test_data::N * sizeof(double)));
  ASSERT_EQ(CUDA_SUCCESS, cudaMalloc(&dexp, sizeof(int32_t)));
  ASSERT_EQ(CUDA_SUCCESS, cudaMemcpy(ddata, hdata.data(), test_data::N * sizeof(double), cudaMemcpyHostToDevice));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  uint32_t nBlocksReduce = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / RDC_THREADS_PER_BLOCK;
  ozigpu::abs_max_nextpow2_reduce<double><<<nBlocksReduce, RDC_THREADS_PER_BLOCK>>>(dexp, ddata, test_data::N);
  for (const auto x : hdata) {
    if (const int32_t e = static_cast<int32_t>(std::ceil(std::log2(std::abs(x)))); e > hexp_real)
      hexp_real = e;
  }
  ASSERT_EQ(CUDA_SUCCESS, cudaDeviceSynchronize());
  ASSERT_EQ(CUDA_SUCCESS, cudaMemcpy(&hexp, dexp, sizeof(int32_t), cudaMemcpyDeviceToHost));
  ASSERT_EQ(hexp_real, hexp);

  cudaFree(ddata);
  cudaFree(dexp);
}

TEST_F(OzakiTestGPU, SliceNormal) {
  std::mt19937 engine{test_data::SEED};
  std::vector<double> hdata(test_data::N_SMALL);
  const int32_t num_slices = 12;
  std::vector<int8_t> hslices(num_slices * test_data::N_SMALL);
  int32_t base_exp;

  std::normal_distribution<double> normal_dist(0.0, 1.0);
  for (size_t i = 0; i < test_data::N_SMALL; ++i) {
    hdata[i] = normal_dist(engine);
  }

  gpu_ozaki_decomposition(hslices.data(), &base_exp, hdata.data(), num_slices, test_data::N_SMALL);
  ASSERT_EQ(CUDA_SUCCESS, cudaPeekAtLastError());

  std::vector<double> hdata_reconstruction(test_data::N, 0.0);
  for (int32_t s = num_slices - 1; s >= 0; --s) {
    int32_t exp = 7 * (s + 1) - base_exp;
    for (uint32_t i = 0; i < 32; ++i)
      for (uint32_t j = 0; j < 32; ++j)
        hdata_reconstruction[i * 32 + j] +=
            std::scalbn(static_cast<double>(hslices[s * test_data::N_SMALL + i * 32 + j]), -exp);
  }
  for (uint32_t i = 0; i < test_data::N_SMALL; ++i)
    ASSERT_EQ(hdata_reconstruction[i], hdata[i]) << "Iter[" << i << "]";
}

TEST_F(OzakiTestGPU, SliceBeta) {
  std::mt19937 engine{test_data::SEED};
  std::vector<double> hdata(test_data::N_SMALL);
  const int32_t num_slices = 32;
  std::vector<int8_t> hslices(num_slices * test_data::N_SMALL);
  int32_t base_exp;

  std::gamma_distribution<double> gamma_dist(0.2, 1.0);
  std::bernoulli_distribution bool_dist(0.5);

  for (size_t i = 0; i < test_data::N_SMALL; ++i) {
    const double g1 = gamma_dist(engine);
    const double g2 = gamma_dist(engine);

    const double beta_val = g1 / (g1 + g2);
    const double scalar =
        bool_dist(engine) ? (1 / std::numeric_limits<double>::epsilon()) : std::numeric_limits<double>::epsilon();
    const double scaled_val = beta_val * scalar;

    hdata[i] = bool_dist(engine) ? scaled_val : -scaled_val;
  }

  gpu_ozaki_decomposition(hslices.data(), &base_exp, hdata.data(), num_slices, test_data::N_SMALL);
  ASSERT_EQ(CUDA_SUCCESS, cudaPeekAtLastError());

  std::vector<double> hdata_reconstruction(test_data::N, 0.0);
  for (int32_t s = num_slices - 1; s >= 0; --s) {
    int32_t exp = 7 * (s + 1) - base_exp;
    for (uint32_t i = 0; i < 32; ++i)
      for (uint32_t j = 0; j < 32; ++j)
        hdata_reconstruction[i * 32 + j] +=
            std::scalbn(static_cast<double>(hslices[s * test_data::N_SMALL + i * 32 + j]), -exp);
  }
  for (uint32_t i = 0; i < test_data::N_SMALL; ++i)
    ASSERT_EQ(hdata_reconstruction[i], hdata[i]) << "Iter[" << i << "]";
}
