#include <gtest/gtest.h>

#include <ozaki/core/stencil.hpp>
#include <ozaki/cpu/ozi.hpp>
#include "../test_utils.hpp"

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

#ifdef OZAKI_HAVE_FLOAT128
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

  auto result_gold = test_utils::compute_gold_standard(stencil, u, {test_data::N});

  // we expect Ozaki to be bit-identical or within 1 ULP of the gold Standard
  for (size_t j = 0; j < test_data::N; ++j) {
    if constexpr (std::is_same_v<Scalar, double>) {
      ASSERT_DOUBLE_EQ(result_ozaki[j], result_gold[j]) << "Mismatch at index " << j;
    } else {
      ASSERT_FLOAT_EQ(result_ozaki[j], result_gold[j]) << "Mismatch at index " << j;
    }
  }
}

TYPED_TEST(OzakiTestCPU, Stencil2D_Laplace) {
  using Scalar = TypeParam;
  constexpr size_t H = 64, W = 64;
  auto u = this->generate_random(H * W);

  // 5-point Laplacian: [0,1,0; 1,-4,1; 0,1,0]
  ozi::Stencil<Scalar, 3, 3> laplacian(
      {Scalar(0), Scalar(1), Scalar(0), Scalar(1), Scalar(-4), Scalar(1), Scalar(0), Scalar(1), Scalar(0)});

  std::vector<Scalar> result_ozaki(H * W);
  ozicpu::stencil<Scalar, 3, 3>(result_ozaki, laplacian, u, {H, W});

  auto result_gold = test_utils::compute_gold_standard(laplacian, u, std::array<size_t, 2>{H, W});

  double rmse = test_utils::interior_relative_rmse<Scalar, 2>(result_ozaki, result_gold, {H, W}, laplacian.offsets);
  EXPECT_LT(rmse, this->get_tolerance()) << "2D Laplacian interior RMSE too high";
}

TYPED_TEST(OzakiTestCPU, Stencil2D_Heat) {
  using Scalar = TypeParam;
  constexpr size_t H = 64, W = 64;
  auto u = this->generate_random(H * W);

  // Heat 3x3: {0, 3/16, 0, 3/16, 4/16, 3/16, 0, 3/16, 0}
  ozi::Stencil<Scalar, 3, 3> heat({Scalar(0), Scalar(3.0 / 16), Scalar(0), Scalar(3.0 / 16), Scalar(4.0 / 16),
                                   Scalar(3.0 / 16), Scalar(0), Scalar(3.0 / 16), Scalar(0)});

  std::vector<Scalar> result_ozaki(H * W);
  ozicpu::stencil<Scalar, 3, 3>(result_ozaki, heat, u, {H, W});

  auto result_gold = test_utils::compute_gold_standard(heat, u, std::array<size_t, 2>{H, W});

  double rmse = test_utils::interior_relative_rmse<Scalar, 2>(result_ozaki, result_gold, {H, W}, heat.offsets);
  EXPECT_LT(rmse, this->get_tolerance()) << "2D Heat interior RMSE too high";
}

TYPED_TEST(OzakiTestCPU, Stencil3D_Box) {
  using Scalar = TypeParam;
  constexpr size_t D = 16, H = 16, W = 16;
  auto u = this->generate_random(D * H * W);

  // Uniform 3x3x3 box filter: all coefficients = 1/27
  ozi::Stencil<Scalar, 3, 3, 3> box;
  for (auto &c : box.coeffs)
    c = Scalar(1.0 / 27);

  std::vector<Scalar> result_ozaki(D * H * W);
  ozicpu::stencil<Scalar, 3, 3, 3>(result_ozaki, box, u, {D, H, W});

  auto result_gold = test_utils::compute_gold_standard(box, u, std::array<size_t, 3>{D, H, W});

  double rmse = test_utils::interior_relative_rmse<Scalar, 3>(result_ozaki, result_gold, {D, H, W}, box.offsets);
  EXPECT_LT(rmse, this->get_tolerance()) << "3D Box interior RMSE too high";
}

TYPED_TEST(OzakiTestCPU, StencilPolynomialExact1D) {
  using Scalar = TypeParam;
  constexpr size_t N = 256;

  // u(x) = x^2, stencil {1, -2, 1} should give 2.0 for interior cells
  auto stencil = test_data::d2x<Scalar>;
  std::vector<Scalar> u(N);
  for (size_t i = 0; i < N; ++i)
    u[i] = static_cast<Scalar>(static_cast<double>(i) * static_cast<double>(i));

  std::vector<Scalar> result_ozaki(N);
  ozicpu::stencil<Scalar, 3>(result_ozaki, stencil, u, {N});

  // Interior cells (indices 1..N-2) should all be 2.0
  for (size_t j = stencil.offsets[0]; j < N - stencil.offsets[0]; ++j) {
    double err = std::abs(static_cast<double>(result_ozaki[j]) - 2.0);
    EXPECT_LT(err, this->get_tolerance()) << "Polynomial exactness failed at index " << j;
  }
}
