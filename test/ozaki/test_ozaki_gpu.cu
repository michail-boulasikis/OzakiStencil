#include "gtest/gtest.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <functional>
#include <gtest/gtest.h>

#include <limits>
#include <ozaki/core/stencil.hpp>
#include <ozaki/gpu/gpu_checks.cuh>
#include <ozaki/gpu/ozi.cuh>

#include <cstdint>
#include <random>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename T> class OzakiTestGPU : public ::testing::Test {
public:
  static constexpr uint64_t SEED = 0x123456789;
  static constexpr T CVAL = static_cast<T>(0xDEADBEEF);
  static constexpr size_t N_SMALL = 1024;
  static constexpr size_t N = N_SMALL * N_SMALL;
  T *h_data, *h_data_s, *d_data, *d_data_s;

protected:
  void generate_normal(T *out, size_t size, double mean = 0.0, double stddev = 1.0) {
    std::mt19937 engine(SEED);
    std::normal_distribution<double> dist(mean, stddev);

    for (size_t i = 0; i < size; ++i)
      out[i] = static_cast<T>(dist(engine));
  }

  double get_tolerance() const {
    if constexpr (std::is_same_v<T, float>) {
      return 1e-6;
    } else {
      return 1e-12;
    }
  }

  void verify_slice_and_mend(int32_t num_slices, const std::function<void()> &datagen) {
    using Scalar = T;

    datagen();
    ASSERT_EQ(CUDA_SUCCESS, cudaMemcpy(this->d_data, this->h_data, this->N * sizeof(Scalar), cudaMemcpyHostToDevice));

    int8_t *d_slices;
    ASSERT_EQ(CUDA_SUCCESS, cudaMalloc(&d_slices, num_slices * this->N * sizeof(int8_t)));

    int32_t *d_ptr = reinterpret_cast<int32_t *>(this->d_data_s);
    ozigpu::fill<int32_t><<<1, 1>>>(d_ptr, std::numeric_limits<int32_t>::min(), 1ull);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint64_t nThreadsMap = std::min(static_cast<uint64_t>(RDC_THREADS_PER_BLOCK), this->N / 4);
    uint32_t nBlocksMap = ((this->N / 4) + nThreadsMap - 1) / nThreadsMap;
    uint32_t nBlocksReduce = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / RDC_THREADS_PER_BLOCK;

    ozigpu::abs_max_nextpow2_reduce<Scalar><<<nBlocksReduce, RDC_THREADS_PER_BLOCK>>>(d_ptr, this->d_data, this->N);
    ozigpu::slice<ozigpu::OzakiFP_Int8<Scalar>>
        <<<nBlocksMap, nThreadsMap>>>(d_slices, this->d_data, d_ptr, num_slices, this->N);
    ASSERT_EQ(CUDA_SUCCESS, cudaDeviceSynchronize());
    ozigpu::mend<ozigpu::OzakiFP_Int8<Scalar>>
        <<<nBlocksMap, nThreadsMap>>>(this->d_data, d_slices, d_ptr, num_slices, this->N);
    ASSERT_EQ(CUDA_SUCCESS, cudaDeviceSynchronize());

    Scalar *h_op_result = new Scalar[this->N];
    ASSERT_EQ(CUDA_SUCCESS, cudaMemcpy(h_op_result, this->d_data, this->N * sizeof(Scalar), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < this->N; ++i)
      ASSERT_EQ(h_op_result[i], this->h_data[i]) << "Iter[" << i << "]";

    cudaFree(d_slices);
  }

  void generate_beta(T *out, size_t size, double a = 0.2, double b = 1.0) {
    std::mt19937 engine{SEED};
    std::gamma_distribution<double> gamma_dist(a, b);
    std::bernoulli_distribution bool_dist(0.5);

    for (size_t i = 0; i < size; ++i) {
      double g1 = gamma_dist(engine);
      double g2 = gamma_dist(engine);
      double beta = g1 / (g1 + g2);
      double eps = static_cast<double>(std::numeric_limits<T>::epsilon());
      double scalar = bool_dist(engine) ? (1.0 / eps) : eps;
      out[i] = static_cast<T>(bool_dist(engine) ? (beta * scalar) : -(beta * scalar));
    }
  }

  void SetUp() override {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No CUDA-capable device found. Skipping test.";
    }

    cudaSetDevice(0);
    h_data = new T[N];
    h_data_s = new T[N_SMALL];
    cudaMalloc(&d_data, sizeof(T) * N);
    cudaMalloc(&d_data_s, sizeof(T) * N_SMALL);
  }

  void TearDown() override {
    delete[] h_data;
    delete[] h_data_s;
    cudaFree(d_data);
    cudaFree(d_data_s);
  }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(OzakiTestGPU, TestTypes);

TYPED_TEST(OzakiTestGPU, Fill) {
  using Scalar = TypeParam;

  ozigpu::fill<Scalar><<<this->N_SMALL, this->N_SMALL>>>(this->d_data, this->CVAL, this->N);

  ASSERT_EQ(CUDA_SUCCESS, cudaDeviceSynchronize());
  ASSERT_EQ(CUDA_SUCCESS, cudaMemcpy(this->h_data, this->d_data, this->N * sizeof(Scalar), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < this->N; ++i)
    ASSERT_EQ(this->CVAL, this->h_data[i]) << "Iter[" << i << "]";
}

TYPED_TEST(OzakiTestGPU, ScaleExtraction) {
  using Scalar = TypeParam;

  this->generate_beta(this->h_data, this->N);
  ASSERT_EQ(CUDA_SUCCESS, cudaMemcpy(this->d_data, this->h_data, this->N * sizeof(Scalar), cudaMemcpyHostToDevice));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  uint32_t nBlocksReduce = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / RDC_THREADS_PER_BLOCK;
  ozigpu::abs_max_nextpow2_reduce<Scalar>
      <<<nBlocksReduce, RDC_THREADS_PER_BLOCK>>>(reinterpret_cast<int32_t *>(this->d_data_s), this->d_data, this->N);

  int32_t hexp_real = std::numeric_limits<int32_t>::min();
  for (size_t i = 0; i < this->N; ++i)
    if (const int32_t e = static_cast<int32_t>(std::ceil(std::log2(std::abs(this->h_data[i])))); e > hexp_real)
      hexp_real = e;

  int32_t *h_ptr = reinterpret_cast<int32_t *>(this->h_data_s);
  ASSERT_EQ(CUDA_SUCCESS, cudaDeviceSynchronize());
  ASSERT_EQ(CUDA_SUCCESS, cudaMemcpy(h_ptr, this->d_data_s, sizeof(int32_t), cudaMemcpyDeviceToHost));
  ASSERT_EQ(hexp_real, *h_ptr);
}

TYPED_TEST(OzakiTestGPU, SliceAndMendNormal) {
  this->verify_slice_and_mend(12, [this]() { this->generate_normal(this->h_data, this->N); });
}

TYPED_TEST(OzakiTestGPU, SliceAndMendBeta) {
  this->verify_slice_and_mend(36, [this]() { this->generate_beta(this->h_data, this->N); });
}
