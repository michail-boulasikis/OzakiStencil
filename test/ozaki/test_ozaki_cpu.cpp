#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <ozaki/core/stencil.hpp>
#include <ozaki/cpu/ozi.hpp>

#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace test_data {
static constexpr uint64_t SEED = 0x12345;
static constexpr size_t N = 1024;

template <typename T> const std::map<std::string, std::vector<T>> get_extremes() {
  return {
      {"min", {std::numeric_limits<T>::min(), static_cast<T>(-2.0), static_cast<T>(1.0)}},
      {"minus_min", {-std::numeric_limits<T>::min(), static_cast<T>(-2.0), static_cast<T>(1.0)}},
      {"eps", {std::numeric_limits<T>::epsilon(), static_cast<T>(-2.0), static_cast<T>(1.0)}},
      {"minus_eps", {-std::numeric_limits<T>::epsilon(), static_cast<T>(-2.0), static_cast<T>(1.0)}},
      {"max", {std::numeric_limits<T>::max(), static_cast<T>(-2.0), static_cast<T>(1.0)}},
      {"minus_max", {-std::numeric_limits<T>::max(), static_cast<T>(-2.0), static_cast<T>(1.0)}},
      // Note: Mixing min/max in one vector might cause precision loss for T = float during summation checks,
      // but is fine for slice/mend integrity.
      {"min_max", {std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), static_cast<T>(1.0)}},
  };
}
template <typename T> const std::vector<T> zeros{0, 0, 0};

template <typename T> const std::vector<T> infty{0, std::numeric_limits<T>::infinity(), 0};

template <typename T>
const ozi::Stencil<T, 3> d2x{{static_cast<T>(1.0), static_cast<T>(-2.0), static_cast<T>(1.0)}, {1}};

} // namespace test_data

template <typename T, typename InputContainer, size_t... Dims>
std::vector<T> compute_gold_standard(const ozi::Stencil<T, Dims...> &stencil, const InputContainer &input,
                                     const std::array<size_t, sizeof...(Dims)> &shape) {

#ifdef __GNUC__
  using HighPrec = __float128;
#else
  using HighPrec = long double;
#endif
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

template <typename T> class OzakiTestCPU : public ::testing::Test {
protected:
  void SetUp() override {}

  double get_tolerance() const {
    if constexpr (std::is_same_v<T, float>)
      return 1e-6;
    else
      return 1e-12;
  }

  std::vector<T> generate_random(size_t size, double mean = 0.0, double stddev = 1.0) {
    std::vector<T> vec(size);
    std::mt19937 engine(test_data::SEED);
    std::normal_distribution<double> dist(mean, stddev);

    for (auto &val : vec) {
      val = static_cast<T>(dist(engine));
    }
    return vec;
  }
};

#ifdef __GNUC__
using TestTypes = ::testing::Types<float, double, long double, __float128>;
#else
using TestTypes = ::testing::Types<float, double, long double>
#endif

TYPED_TEST_SUITE(OzakiTestCPU, TestTypes);
TYPED_TEST(OzakiTestCPU, SliceAndMendNormal) {
  using Scalar = TypeParam;
  auto tcase = this->generate_random(test_data::N);

  auto slices = ozicpu::slice<ozicpu::OzakiFP_Int8<Scalar>>(tcase);

  std::vector<Scalar> recovered(tcase.size());
  ozicpu::mend<ozicpu::OzakiFP_Int8<Scalar>>(recovered, slices);

  for (size_t i = 0; i < tcase.size(); ++i) {
    ASSERT_EQ(recovered[i], tcase[i]) << "Mismatch at index " << i;
  }
}

TYPED_TEST(OzakiTestCPU, SliceAndMendExtremes) {
  using Scalar = TypeParam;
  const auto cases = test_data::get_extremes<Scalar>();

  for (const auto &[name, data] : cases) {
    auto slices = ozicpu::slice<ozicpu::OzakiFP_Int8<Scalar>>(data);
    std::vector<Scalar> recovered(data.size());
    ozicpu::mend<ozicpu::OzakiFP_Int8<Scalar>>(recovered, slices);

    for (size_t i = 0; i < data.size(); ++i) {
      ASSERT_EQ(recovered[i], data[i]) << "Failed case: " << name;
    }
  }
}

TYPED_TEST(OzakiTestCPU, SliceAndMendZeros) {
  using Scalar = TypeParam;
  auto data = test_data::zeros<Scalar>;

  auto slices = ozicpu::slice<ozicpu::OzakiFP_Int8<Scalar>>(data);
  std::vector<Scalar> recovered(data.size());
  ozicpu::mend<ozicpu::OzakiFP_Int8<Scalar>>(recovered, slices);

  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(recovered[i], data[i]);
  }
  ASSERT_EQ(slices.values.size(), 0) << "Zero vector should produce 0 slices";
}

TYPED_TEST(OzakiTestCPU, InfinityThrows) {
  using Scalar = TypeParam;
  if constexpr (std::numeric_limits<Scalar>::has_infinity) {
    auto data = test_data::infty<Scalar>;
    ASSERT_THROW({ ozicpu::slice<ozicpu::OzakiFP_Int8<Scalar>>(data); }, std::runtime_error);
  }
}

TYPED_TEST(OzakiTestCPU, MatMulNormal) {
  using Scalar = TypeParam;
  using MapMat = Eigen::Map<Eigen::Matrix<Scalar, 32, 32, Eigen::ColMajor>>;

  auto vecA = this->generate_random(test_data::N);
  auto vecB = this->generate_random(test_data::N);
  std::vector<Scalar> vecC_ozaki(test_data::N);

  MapMat A(vecA.data());
  MapMat B(vecB.data());
  Eigen::Matrix<Scalar, 32, 32, Eigen::ColMajor> C_actual = A * B;

  auto slicesA = ozicpu::slice<ozicpu::OzakiFP_Int8<Scalar>>(vecA);
  auto slicesB = ozicpu::slice<ozicpu::OzakiFP_Int8<Scalar>>(vecB);
  ozicpu::mul<ozicpu::OzakiFP_Int8<Scalar>>(vecC_ozaki, slicesA, slicesB, 32, 32, 32);

  MapMat C_ozaki(vecC_ozaki.data());

  ASSERT_TRUE(C_actual.isApprox(C_ozaki, this->get_tolerance()));
}

TYPED_TEST(OzakiTestCPU, MatMulBeta) {
  using Scalar = TypeParam;
  using MapMat = Eigen::Map<Eigen::Matrix<Scalar, 32, 32, Eigen::ColMajor>>;

  std::mt19937 engine{test_data::SEED};
  std::gamma_distribution<double> gamma_dist(0.2, 1.0);
  std::bernoulli_distribution bool_dist(0.5);

  auto betadist = [&]() -> Scalar {
    double g1 = gamma_dist(engine);
    double g2 = gamma_dist(engine);
    double beta = g1 / (g1 + g2);

    double eps = static_cast<double>(std::numeric_limits<Scalar>::epsilon());
    double scalar = bool_dist(engine) ? (1.0 / eps) : eps;

    return static_cast<Scalar>(bool_dist(engine) ? (beta * scalar) : -(beta * scalar));
  };

  std::vector<Scalar> vecA(test_data::N), vecB(test_data::N), vecC_ozaki(test_data::N);
  for (size_t i = 0; i < test_data::N; ++i) {
    vecA[i] = betadist();
    vecB[i] = betadist();
  }

  MapMat A(vecA.data());
  MapMat B(vecB.data());
  auto C_actual = A * B;

  auto slicesA = ozicpu::slice<ozicpu::OzakiFP_Int8<Scalar>>(vecA);
  auto slicesB = ozicpu::slice<ozicpu::OzakiFP_Int8<Scalar>>(vecB);
  ozicpu::mul<ozicpu::OzakiFP_Int8<Scalar>>(vecC_ozaki, slicesA, slicesB, 32, 32, 32);

  MapMat C_ozaki(vecC_ozaki.data());
  ASSERT_TRUE(C_actual.isApprox(C_ozaki, this->get_tolerance()));
}

TYPED_TEST(OzakiTestCPU, MatMulScaleSkew) {
  // large * tiny ~ unity
  using Scalar = TypeParam;
  using MapMat = Eigen::Map<Eigen::Matrix<Scalar, 32, 32, Eigen::ColMajor>>;

  std::mt19937 engine{test_data::SEED};
  double huge_base = std::is_same_v<Scalar, float> ? 1.0e30 : 1.0e100;
  double tiny_base = std::is_same_v<Scalar, float> ? 1.0e-30 : 1.0e-100;

  std::uniform_real_distribution<double> dist_huge(huge_base, huge_base * 5.0);
  std::uniform_real_distribution<double> dist_tiny(tiny_base, tiny_base * 5.0);

  std::vector<Scalar> vecA(test_data::N), vecB(test_data::N), vecC_ozaki(test_data::N);
  for (size_t i = 0; i < test_data::N; ++i) {
    vecA[i] = static_cast<Scalar>(dist_huge(engine));
    vecB[i] = static_cast<Scalar>(dist_tiny(engine));
  }

  MapMat A(vecA.data());
  MapMat B(vecB.data());
  auto C_actual = A * B;

  auto slicesA = ozicpu::slice<ozicpu::OzakiFP_Int8<Scalar>>(vecA);
  auto slicesB = ozicpu::slice<ozicpu::OzakiFP_Int8<Scalar>>(vecB);
  ozicpu::mul<ozicpu::OzakiFP_Int8<Scalar>>(vecC_ozaki, slicesA, slicesB, 32, 32, 32);
  MapMat C_ozaki(vecC_ozaki.data());

  double tol = std::is_same_v<Scalar, float> ? 1e-5 : 1e-11;
  ASSERT_TRUE(C_actual.isApprox(C_ozaki, tol));
}

TYPED_TEST(OzakiTestCPU, MatMulLogUniform) {
  using Scalar = TypeParam;
  using MapMat = Eigen::Map<Eigen::Matrix<Scalar, 32, 32, Eigen::ColMajor>>;

  std::mt19937 engine{test_data::SEED};
  double log_range = std::is_same_v<Scalar, float> ? 5.0 : 10.0;

  std::uniform_real_distribution<double> exp_dist(-log_range, log_range);
  std::uniform_real_distribution<double> man_dist(-1.0, 1.0);
  std::normal_distribution<double> norm_dist(0.0, 1.0);

  std::vector<Scalar> vecA(test_data::N), vecB(test_data::N), vecC_ozaki(test_data::N);
  for (size_t i = 0; i < test_data::N; ++i) {
    vecA[i] = static_cast<Scalar>(man_dist(engine) * std::pow(10.0, exp_dist(engine)));
    vecB[i] = static_cast<Scalar>(norm_dist(engine));
  }

  MapMat A(vecA.data());
  MapMat B(vecB.data());
  auto C_actual = A * B;

  auto slicesA = ozicpu::slice<ozicpu::OzakiFP_Int8<Scalar>>(vecA);
  auto slicesB = ozicpu::slice<ozicpu::OzakiFP_Int8<Scalar>>(vecB);
  ozicpu::mul<ozicpu::OzakiFP_Int8<Scalar>>(vecC_ozaki, slicesA, slicesB, 32, 32, 32);
  MapMat C_ozaki(vecC_ozaki.data());

  ASSERT_TRUE(C_actual.isApprox(C_ozaki, this->get_tolerance()));
}

TYPED_TEST(OzakiTestCPU, MatMulRectangular) {
  using Scalar = TypeParam;
  using MapMat = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>;

  size_t M = 16, K = 8, N = 64;

  auto vecA = this->generate_random(M * K);
  auto vecB = this->generate_random(K * N);
  std::vector<Scalar> vecC_ozaki(M * N);

  MapMat A(vecA.data(), M, K);
  MapMat B(vecB.data(), K, N);
  auto C_actual = A * B;

  auto slicesA = ozicpu::slice<ozicpu::OzakiFP_Int8<Scalar>>(vecA);
  auto slicesB = ozicpu::slice<ozicpu::OzakiFP_Int8<Scalar>>(vecB);
  ozicpu::mul<ozicpu::OzakiFP_Int8<Scalar>>(vecC_ozaki, slicesA, slicesB, M, N, K);

  MapMat C_ozaki(vecC_ozaki.data(), M, N);

  ASSERT_TRUE(C_actual.isApprox(C_ozaki, this->get_tolerance()));
}

// verify that Ozaki survives catastrophic cancellation where Naive fails
TYPED_TEST(OzakiTestCPU, StabilityCancellation) {
  using Scalar = TypeParam;

  if constexpr (std::is_same_v<Scalar, double>) { // test only valid when using double precision
    ozi::Stencil<Scalar, 3> sum_stencil({static_cast<Scalar>(1.0), static_cast<Scalar>(1.0), static_cast<Scalar>(1.0)});

    // Naive Math: (1e16 + 1.0) - 1e16  ->  1e16 - 1e16 -> 0.0 (The 1.0 is lost)
    // Exact Math: 1e16 + 1.0 - 1e16 -> 1.0
    Scalar huge = static_cast<Scalar>(1e16);
    std::vector<Scalar> input = {huge, static_cast<Scalar>(1.0), -huge};
    std::vector<Scalar> result_ozaki(3);
    std::vector<Scalar> result_naive(3);

    ozicpu::stencil<Scalar, 3>(result_ozaki, sum_stencil, input, {3});
    sum_stencil.template apply_to<Scalar, std::vector<Scalar>, std::vector<Scalar>>(result_naive, input, {3});

    Scalar expected = static_cast<Scalar>(1.0);
    ASSERT_NEAR(result_ozaki[1], expected, 1e-5) << "Ozaki failed to handle cancellation.";
    EXPECT_NE(result_naive[1], expected)
      << "Warning: Naive implementation accidentally got the right answer (likely due to FMA or compiler opts).";
    } else {
      GTEST_SKIP();
    }
}

TYPED_TEST(OzakiTestCPU, SimpleStencil) {
  using Scalar = TypeParam;

  auto u = this->generate_random(test_data::N);
  auto stencil = test_data::d2x<Scalar>;
  std::vector<Scalar> result_ozaki(u.size());

  ozicpu::stencil<Scalar, 3>(result_ozaki, stencil, u, {test_data::N});

  auto result_gold = compute_gold_standard(stencil, u, {test_data::N});

  // we expect Ozaki to be bit-identical or within 1 ULP of the gold Standard
  for (size_t j = 0; j < test_data::N; ++j) {
    if constexpr (std::is_same_v<Scalar, double>) {
      ASSERT_DOUBLE_EQ(result_ozaki[j], result_gold[j]) << "Mismatch at index " << j;
    } else {
      ASSERT_FLOAT_EQ(result_ozaki[j], result_gold[j]) << "Mismatch at index " << j;
    }
  }
}
