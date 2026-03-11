#include <gtest/gtest.h>
#include <concepts>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <ios>
#include <ozaki/core/stencil.hpp>
#include <ozaki/gpu/ozi_kernels.cuh>
#include "../test_utils.hpp"

#include <bit>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename T> int32_t get_slice_max_normal() {
  if constexpr (std::is_same_v<T, float>) {
    return 9;
  } else if constexpr (std::is_same_v<T, double>) {
    return 12;
  } else {
    return 24;
  }
}

template <typename T> int32_t get_slice_max_beta() {
  if constexpr (std::is_same_v<T, float>) {
    return 24;
  } else if constexpr (std::is_same_v<T, double>) {
    return 48;
  } else {
    return 96;
  }
}

template <typename T> class OzakiTestGPU : public ::testing::Test {
public:
  static constexpr uint64_t SEED = 0x123456789;
  static constexpr size_t N_SMALL = 1024;
  static constexpr size_t N = N_SMALL * N_SMALL;
  T *h_data, *d_data;
  ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<T>> *d_desc = nullptr;
  ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<T>> h_desc;

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

  void verify_slice_and_mend(int32_t num_slices_max, bool dynamic_slicing, const std::function<void()> &datagen) {
    using Scalar = T;

    datagen();
    ASSERT_EQ(CUDA_SUCCESS, cudaMemcpy(this->d_data, this->h_data, this->N * sizeof(Scalar), cudaMemcpyHostToDevice));

    int8_t *d_slices;
    ASSERT_EQ(CUDA_SUCCESS,
              cudaMalloc(&d_slices, (num_slices_max + 1) * this->N * sizeof(int8_t))); // + 1 for the carry slice

    h_desc = ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Scalar>>();
    if (dynamic_slicing)
      h_desc.n_slices_input = 0;
    else
      h_desc.n_slices_input = num_slices_max;
    ASSERT_EQ(CUDA_SUCCESS, cudaMemcpy(d_desc, &h_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Scalar>>),
                                       cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint64_t nThreadsMap = std::min(static_cast<uint64_t>(RDC_THREADS_PER_BLOCK), this->N / 4);
    uint32_t nBlocksMap = ((this->N / 4) + nThreadsMap - 1) / nThreadsMap;
    uint32_t nBlocksReduce = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / RDC_THREADS_PER_BLOCK;

    ozigpu::dyn_range<<<nBlocksReduce, RDC_THREADS_PER_BLOCK>>>(this->d_desc, this->d_data, this->N);
    ozigpu::slice<ozigpu::OzakiFP_Int8<Scalar>>
        <<<nBlocksMap, nThreadsMap>>>(d_slices, this->d_data, this->d_desc, this->N);
    ASSERT_EQ(CUDA_SUCCESS, cudaDeviceSynchronize());
    ozigpu::mend<ozigpu::OzakiFP_Int8<Scalar>>
        <<<nBlocksMap, nThreadsMap>>>(this->d_data, d_slices, this->d_desc, this->N);
    ASSERT_EQ(CUDA_SUCCESS, cudaDeviceSynchronize());

    Scalar *h_op_result = new Scalar[this->N];
    ASSERT_EQ(CUDA_SUCCESS, cudaMemcpy(h_op_result, this->d_data, this->N * sizeof(Scalar), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < this->N; ++i)
      ASSERT_EQ(h_op_result[i], this->h_data[i]) << "Iter[" << i << "]";

    delete[] h_op_result;
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
    cudaMalloc(&d_data, sizeof(T) * N);
    cudaMalloc(&d_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<T>>));
  }

  void TearDown() override {
    delete[] h_data;
    cudaFree(d_data);
    cudaFree(d_desc);
  }
};

#ifdef OZAKI_HAVE_FLOAT128
using TestTypes = ::testing::Types<float, double, __float128>;
#else
using TestTypes = ::testing::Types<float, double>;
#endif

TYPED_TEST_SUITE(OzakiTestGPU, TestTypes);

TYPED_TEST(OzakiTestGPU, ScaleExtraction) {
  using Scalar = TypeParam;

  this->generate_beta(this->h_data, this->N);
  ASSERT_EQ(CUDA_SUCCESS, cudaMemcpy(this->d_data, this->h_data, this->N * sizeof(Scalar), cudaMemcpyHostToDevice));

  this->h_desc = ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Scalar>>();
  ASSERT_EQ(CUDA_SUCCESS,
            cudaMemcpy(this->d_desc, &this->h_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Scalar>>),
                       cudaMemcpyHostToDevice));
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  uint32_t nBlocksReduce = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / RDC_THREADS_PER_BLOCK;
  ozigpu::dyn_range<<<nBlocksReduce, RDC_THREADS_PER_BLOCK>>>(this->d_desc, this->d_data, this->N);

  int32_t hexp_real_max = std::numeric_limits<int32_t>::min();
  for (size_t i = 0; i < this->N; ++i) {
    if constexpr (std::is_same_v<Scalar, __float128>) {
      if (const int32_t e = static_cast<int32_t>(ceilq(log2q(fabsq(this->h_data[i])))); e > hexp_real_max) {
        hexp_real_max = e;
      }
    } else {
      if (const int32_t e = static_cast<int32_t>(std::ceil(std::log2(std::abs(this->h_data[i])))); e > hexp_real_max)
        hexp_real_max = e;
    }
  }

  ASSERT_EQ(CUDA_SUCCESS, cudaDeviceSynchronize());
  ASSERT_EQ(CUDA_SUCCESS,
            cudaMemcpy(&this->h_desc, this->d_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Scalar>>),
                       cudaMemcpyDeviceToHost));
  ASSERT_EQ(CUDA_SUCCESS, cudaDeviceSynchronize());
  ASSERT_EQ(hexp_real_max, this->h_desc.e_max);
}

TYPED_TEST(OzakiTestGPU, SliceAndMendNormal) {
  using Scalar = TypeParam;
  // Test both static and dynamic slicing in a single test
  this->verify_slice_and_mend(get_slice_max_normal<Scalar>(), false,
                              [this]() { this->generate_normal(this->h_data, this->N); });
  this->verify_slice_and_mend(get_slice_max_normal<Scalar>(), true,
                              [this]() { this->generate_normal(this->h_data, this->N); });
}

TYPED_TEST(OzakiTestGPU, SliceAndMendBeta) {
  using Scalar = TypeParam;
  // Test both static and dynamic slicing in a single test
  this->verify_slice_and_mend(get_slice_max_beta<Scalar>(), false,
                              [this]() { this->generate_beta(this->h_data, this->N); });
  this->verify_slice_and_mend(get_slice_max_beta<Scalar>(), true,
                              [this]() { this->generate_beta(this->h_data, this->N); });
}
