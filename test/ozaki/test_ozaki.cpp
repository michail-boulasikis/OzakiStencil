#include <gtest/gtest.h>
#include <limits>
#include <ozaki/core/ozi.hpp>
#include <ozaki/core/stencil.hpp>
#include <random>
#include <stdexcept>
#include <vector>
#include <ozaki/gpu/kernels.cuh>

namespace test_data {
static constexpr uint64_t SEED = 0x123456789;
static constexpr uint64_t N = 1024;
const std::map<std::string, std::vector<double>> extreme{
    {"min", {{std::numeric_limits<double>::min(), -2.0, 1.0}}},
    {"minus_min", {{-std::numeric_limits<double>::min(), -2.0, 1.0}}},
    {"eps", {{std::numeric_limits<double>::epsilon(), -2.0, 1.0}}},
    {"minus_eps", {{-std::numeric_limits<double>::epsilon(), -2.0, 1.0}}},
    {"max", {{std::numeric_limits<double>::max(), -2.0, 1.0}}},
    {"minus_max", {{-std::numeric_limits<double>::max(), -2.0, 1.0}}},
    {"min_max", {{std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), 1.0}}},
    {"minus_min_max", {{-std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), 1.0}}},
    {"min_minus_max", {{std::numeric_limits<double>::min(), -std::numeric_limits<double>::max(), 1.0}}},
    {"minus_min_minus_max", {{-std::numeric_limits<double>::min(), -std::numeric_limits<double>::max(), 1.0}}},
};
const std::vector<double> zeros{0.0, 0.0, 0.0};
const std::vector<double> infty{0.0, std::numeric_limits<double>::infinity(), 0.0};
} // namespace test_data

class OzakiOne : public ::testing::Test {
protected:
  void SetUp() override {}
};

class OzakiOneExtremesTest : public ::testing::TestWithParam<std::pair<const std::string, std::vector<double>>> {};

TEST_F(OzakiOne, NormalDist) {
  std::mt19937 engine{test_data::SEED};
  std::vector<double> tcase(test_data::N);

  std::normal_distribution<double> normal_dist(0.0, 1.0);

  for (size_t i = 0; i < test_data::N; ++i) {
    tcase[i] = normal_dist(engine);
  }

  ozaki::OzakiIMatrix<double, int8_t> ozs{tcase, tcase.size()};
  const auto back = ozs.recombine();
  for (size_t i = 0; i < tcase.size(); ++i) {
    ASSERT_EQ(back[i], tcase[i]);
  }
}

TEST_F(OzakiOne, BetaDist) {
  std::mt19937 engine{test_data::SEED};
  std::vector<double> tcase(test_data::N);

  std::gamma_distribution<double> gamma_dist(0.2, 1.0);
  std::bernoulli_distribution bool_dist(0.5);

  for (size_t i = 0; i < test_data::N; ++i) {
    const double g1 = gamma_dist(engine);
    const double g2 = gamma_dist(engine);

    const double beta_val = g1 / (g1 + g2);
    const double scalar =
        bool_dist(engine) ? (1 / std::numeric_limits<double>::epsilon()) : std::numeric_limits<double>::epsilon();
    const double scaled_val = beta_val * scalar;

    tcase[i] = bool_dist(engine) ? scaled_val : -scaled_val;
  }

  ozaki::OzakiIMatrix<double, int8_t> ozs{tcase, tcase.size()};
  const auto back = ozs.recombine();
  for (size_t i = 0; i < tcase.size(); ++i) {
    ASSERT_EQ(back[i], tcase[i]);
  }
}

TEST_P(OzakiOneExtremesTest, HandlesExtremeValues) {
  const auto &param = GetParam();
  const std::string &msg = param.first;
  const std::vector<double> &tcase = param.second;

  const ozaki::OzakiIMatrix<double, int8_t> ozs{tcase, tcase.size()};
  const auto back = ozs.recombine();
  for (size_t i = 0; i < tcase.size(); ++i) {
    ASSERT_EQ(back[i], tcase[i]) << msg;
  }
}

INSTANTIATE_TEST_SUITE_P(AllExtremes, OzakiOneExtremesTest, ::testing::ValuesIn(test_data::extreme),
                         [](const ::testing::TestParamInfo<OzakiOneExtremesTest::ParamType> &info) {
                           return info.param.first;
                         });

TEST_F(OzakiOne, Zeros) {
  const ozaki::OzakiIMatrix<double, int8_t> ozs{test_data::zeros, test_data::zeros.size()};
  const auto back = ozs.recombine();
  for (size_t i = 0; i < test_data::zeros.size(); ++i) {
    ASSERT_EQ(back[i], test_data::zeros[i]);
  }
  ASSERT_EQ(ozs.slices.size(), 0);
}

TEST_F(OzakiOne, WorstCase) {
  std::vector<double> tcase{};
  double a = std::numeric_limits<double>::max();
  // NOTE, this actually skips some close-to-subnormal numbers that cause my implementation to fail.
  // Not sure if I want to support this range, so for now I leave it like that.
  while (std::isnormal(a)) {
    tcase.push_back(a);
    a /= 8;
  }
  ozaki::OzakiIMatrix<double, int8_t> ozs{tcase, tcase.size()};
  const auto back = ozs.recombine();
  for (size_t i = 0; i < tcase.size(); ++i) {
    ASSERT_EQ(back[i], tcase[i]) << "Iter[" << i << "]";
  }
}

TEST_F(OzakiOne, InftyThrows) {
  auto should_throw = [&]() {
    const ozaki::OzakiIMatrix<double, int8_t> ozs{test_data::infty, test_data::infty.size()};
    const auto back = ozs.recombine();
  };
  ASSERT_THROW(should_throw(), std::runtime_error);
}
