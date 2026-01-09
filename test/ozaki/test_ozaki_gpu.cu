#include "gtest/gtest.h"
#include <concepts>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gtest/gtest.h>

#include <ozaki/core/stencil.hpp>
#include <ozaki/gpu/gpu_checks.cuh>
#include <ozaki/gpu/ozi.cuh>

#include <bit>
#include <cstdint>
#include <format>
#include <functional>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename T> struct get_uint_same_size {
  // Ensure we only pass floating point types
  static_assert(std::is_floating_point_v<T>, "Type must be floating point");

  using type =
      typename std::conditional<sizeof(T) == sizeof(std::uint32_t), std::uint32_t,
                                typename std::conditional<sizeof(T) == sizeof(std::uint64_t), std::uint64_t,
                                                          typename std::conditional<sizeof(T) == sizeof(std::uint16_t),
                                                                                    std::uint16_t, // For half-floats
                                                                                    void>::type>::type>::type;

  // Ensure we actually found a match
  static_assert(!std::is_void_v<type>, "No uint type of same size found for this float");
};
template <typename T> using get_uint_same_size_t = typename get_uint_same_size<T>::type;
template <typename T, typename InputContainer, size_t... Dims>
std::vector<T> compute_gold_standard(const ozi::Stencil<T, Dims...> &stencil, const InputContainer &input,
                                     const std::array<size_t, sizeof...(Dims)> &shape) {

  using HighPrec = long double;
  ozi::Stencil<HighPrec, Dims...> s_high;

  s_high.offsets = stencil.offsets;
  for (size_t i = 0; i < stencil.coeffs.size(); ++i) {
    s_high.coeffs[i] = static_cast<HighPrec>(stencil.coeffs[i]);
  }

  std::vector<HighPrec> in_high(input.begin(), input.end());
  std::vector<HighPrec> out_high(input.size());

  s_high.template apply_to<HighPrec>(out_high, in_high, shape);

  std::vector<T> result(out_high.size());
  for (size_t i = 0; i < out_high.size(); ++i) {
    result[i] = static_cast<T>(out_high[i]);
  }

  return result;
}

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

  void verify_slice_and_mend(int32_t num_slices_max, bool dynamic_slicing, const std::function<void()> &datagen) {
    using Scalar = T;

    datagen();
    ASSERT_EQ(CUDA_SUCCESS, cudaMemcpy(this->d_data, this->h_data, this->N * sizeof(Scalar), cudaMemcpyHostToDevice));

    int8_t *d_slices;
    ASSERT_EQ(CUDA_SUCCESS, cudaMalloc(&d_slices, num_slices_max * this->N * sizeof(int8_t)));

    int32_t *d_ptr = reinterpret_cast<int32_t *>(this->d_data_s);
    ozigpu::fill<int32_t><<<1, 1>>>(d_ptr, std::numeric_limits<int32_t>::min(), 1ull);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint64_t nThreadsMap = std::min(static_cast<uint64_t>(RDC_THREADS_PER_BLOCK), this->N / 4);
    uint32_t nBlocksMap = ((this->N / 4) + nThreadsMap - 1) / nThreadsMap;
    uint32_t nBlocksReduce = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / RDC_THREADS_PER_BLOCK;

    ozigpu::dyn_range<Scalar><<<nBlocksReduce, RDC_THREADS_PER_BLOCK>>>(d_ptr, this->d_data, this->N);
    ozigpu::slice<ozigpu::OzakiFP_Int8<Scalar>>
        <<<nBlocksMap, nThreadsMap>>>(d_slices, this->d_data, d_ptr, dynamic_slicing ? 0 : num_slices_max, this->N);
    ASSERT_EQ(CUDA_SUCCESS, cudaDeviceSynchronize());
    ozigpu::mend<ozigpu::OzakiFP_Int8<Scalar>>
        <<<nBlocksMap, nThreadsMap>>>(this->d_data, d_slices, d_ptr, dynamic_slicing ? 0 : num_slices_max, this->N);
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
  ozigpu::dyn_range<Scalar>
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
  this->verify_slice_and_mend(12, false, [this]() { this->generate_normal(this->h_data, this->N); });
}

TYPED_TEST(OzakiTestGPU, SliceAndMendBeta) {
  this->verify_slice_and_mend(36, false, [this]() { this->generate_beta(this->h_data, this->N); });
}

TYPED_TEST(OzakiTestGPU, SliceAndMendNormalDyn) {
  this->verify_slice_and_mend(12, true, [this]() { this->generate_normal(this->h_data, this->N); });
}

TYPED_TEST(OzakiTestGPU, SliceAndMendBetaDyn) {
  this->verify_slice_and_mend(36, true, [this]() { this->generate_beta(this->h_data, this->N); });
}

TYPED_TEST(OzakiTestGPU, OzakiStencil) {
  using Scalar = TypeParam;
  std::vector<Scalar> h_data(this->N);
  this->generate_normal(h_data.data(), this->N);
  std::vector<Scalar> h_ozaki(this->N);

  ozi::Stencil<int8_t, 3> s{{16, 32, 16}, {1}};
  ozi::Stencil<Scalar, 3> s_real{{0.125, 0.25, 0.125}, {1}};

  const int32_t num_slices = 12;
  ASSERT_EQ(CUDA_SUCCESS, cudaMemcpy(this->d_data, h_data.data(), this->N * sizeof(Scalar), cudaMemcpyHostToDevice));

  int8_t *d_slices_in, *d_slices_out;
  ASSERT_EQ(CUDA_SUCCESS, cudaMalloc(&d_slices_in, num_slices * this->N * sizeof(int8_t)));
  ASSERT_EQ(CUDA_SUCCESS, cudaMalloc(&d_slices_out, num_slices * this->N * sizeof(int8_t)));

  int32_t *d_ptr = reinterpret_cast<int32_t *>(this->d_data_s);
  ozigpu::fill<int32_t><<<1, 1>>>(d_ptr, std::numeric_limits<int32_t>::min(), 1ull);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  uint32_t nBlocksReduce = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / RDC_THREADS_PER_BLOCK;

  uint64_t nThreadsMap = std::min(static_cast<uint64_t>(RDC_THREADS_PER_BLOCK), this->N / 4);
  uint32_t nBlocksMap = ((this->N / 4) + nThreadsMap - 1) / nThreadsMap;
  uint64_t nThreadsStencil = 512;
  uint64_t nBlocksStencil = (this->N + nThreadsStencil - 1) / nThreadsStencil;

  ozigpu::dyn_range<Scalar><<<nBlocksReduce, RDC_THREADS_PER_BLOCK>>>(d_ptr, this->d_data, this->N);
  ozigpu::slice<ozigpu::OzakiFP_Int8<Scalar>>
      <<<nBlocksMap, nThreadsMap>>>(d_slices_in, this->d_data, d_ptr, num_slices, this->N);

  ozigpu::stencil_v1<ozigpu::OzakiFP_Int8<Scalar>, 3>
      <<<nBlocksStencil, nThreadsStencil>>>(d_slices_out, d_slices_in, s, {this->N}, num_slices);
  ASSERT_EQ(CUDA_SUCCESS, cudaDeviceSynchronize());

  ozigpu::mend<ozigpu::OzakiFP_Int8<Scalar>>
      <<<nBlocksMap, nThreadsMap>>>(this->d_data, d_slices_out, d_ptr, num_slices, this->N);

  auto h_ref = compute_gold_standard(s_real, h_data, {this->N});

  ASSERT_EQ(CUDA_SUCCESS, cudaMemcpy(h_ozaki.data(), this->d_data, this->N * sizeof(Scalar), cudaMemcpyDeviceToHost));

  for (size_t i = 1; i < this->N - 1; ++i) {
    EXPECT_NEAR(h_ozaki[i], h_ref[i], this->get_tolerance()) << "Ozaki deviated from the gold standard.";
  }
  cudaFree(d_slices_in);
  cudaFree(d_slices_out);
}
