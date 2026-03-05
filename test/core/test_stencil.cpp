#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <ozaki/core/stencil.hpp>
#include <ozaki/cpu/ozi.hpp>

namespace test_data {
static constexpr size_t N = 1024;
static constexpr size_t T = 10;
static constexpr ozi::Stencil<double, 3> d2x{{1.0, -2.0, 1.0}, {1}};
static constexpr ozi::Stencil<double, 3, 3> laplace{{0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0}, {1, 1}};
static constexpr ozi::Stencil<double, 3> impulse{{0.0, 1.0, 0.0}, {1}};
} // namespace test_data

class StencilTests : public ::testing::Test {
protected:
  std::vector<double> x;

  void SetUp() override {
    x.resize(test_data::N);
    std::iota(x.begin(), x.end(), 0.0);
  }
};

TEST_F(StencilTests, AbsoluteIndexing1D) {
  ASSERT_EQ(test_data::impulse[0], 0.0);
  ASSERT_EQ(test_data::impulse[1], 1.0);
  ASSERT_EQ(test_data::impulse[2], 0.0);
  ASSERT_EQ(test_data::impulse(0), 0.0);
  ASSERT_EQ(test_data::impulse(1), 1.0);
  ASSERT_EQ(test_data::impulse(2), 0.0);
}

TEST_F(StencilTests, RelativeIndexing1D) {
  ASSERT_EQ(test_data::impulse.cfrel(-1), 0.0);
  ASSERT_EQ(test_data::impulse.cfrel(0), 1.0);
  ASSERT_EQ(test_data::impulse.cfrel(1), 0.0);
}

TEST_F(StencilTests, AbsoluteIndexing2D) {
  ASSERT_EQ(test_data::laplace[0], 0.0);
  ASSERT_EQ(test_data::laplace[1], 1.0);
  ASSERT_EQ(test_data::laplace[2], 0.0);
  ASSERT_EQ(test_data::laplace[3], 1.0);
  ASSERT_EQ(test_data::laplace[4], -4.0);
  ASSERT_EQ(test_data::laplace[5], 1.0);
  ASSERT_EQ(test_data::laplace[6], 0.0);
  ASSERT_EQ(test_data::laplace[7], 1.0);
  ASSERT_EQ(test_data::laplace[8], 0.0);

  ASSERT_EQ(test_data::laplace(0, 0), 0.0);
  ASSERT_EQ(test_data::laplace(0, 1), 1.0);
  ASSERT_EQ(test_data::laplace(0, 2), 0.0);
  ASSERT_EQ(test_data::laplace(1, 0), 1.0);
  ASSERT_EQ(test_data::laplace(1, 1), -4.0);
  ASSERT_EQ(test_data::laplace(1, 2), 1.0);
  ASSERT_EQ(test_data::laplace(2, 0), 0.0);
  ASSERT_EQ(test_data::laplace(2, 1), 1.0);
  ASSERT_EQ(test_data::laplace(2, 2), 0.0);
}

TEST_F(StencilTests, RelativeIndexing2D) {
  ASSERT_EQ(test_data::laplace.cfrel(-1, -1), 0.0);
  ASSERT_EQ(test_data::laplace.cfrel(-1, 0), 1.0);
  ASSERT_EQ(test_data::laplace.cfrel(-1, 1), 0.0);
  ASSERT_EQ(test_data::laplace.cfrel(0, -1), 1.0);
  ASSERT_EQ(test_data::laplace.cfrel(0, 0), -4.0);
  ASSERT_EQ(test_data::laplace.cfrel(0, 1), 1.0);
  ASSERT_EQ(test_data::laplace.cfrel(1, -1), 0.0);
  ASSERT_EQ(test_data::laplace.cfrel(1, 0), 1.0);
  ASSERT_EQ(test_data::laplace.cfrel(1, 1), 0.0);
}

TEST_F(StencilTests, SelfCorrelationOfImpulseIsImpulse) {
  auto impulse_2 = correlate(test_data::impulse, test_data::impulse);
  ASSERT_EQ(impulse_2.size, 5);
  ASSERT_EQ(impulse_2.cfrel(0), 1.0);
  impulse_2.cfrel(0) = 0.0;
  ASSERT_TRUE(std::all_of(impulse_2.coeffs.begin(), impulse_2.coeffs.end(), [](const double x) { return x == 0.0; }));
}

TEST_F(StencilTests, SelfCorrelationOfD2XIsD4X) {
  auto d4x = correlate(test_data::d2x, test_data::d2x);
  ASSERT_EQ(d4x.cfrel(-2), 1.0);
  ASSERT_EQ(d4x.cfrel(-1), -4.0);
  ASSERT_EQ(d4x.cfrel(0), 6.0);
  ASSERT_EQ(d4x.cfrel(1), -4.0);
  ASSERT_EQ(d4x.cfrel(2), 1.0);
}

TEST_F(StencilTests, ISL) {
  std::vector<double> x_next(this->x.begin(), this->x.end());
  std::vector<double> u(this->x.begin(), this->x.end());
  std::vector<double> u_next(this->x.begin(), this->x.end());

  for (size_t i = 0; i < test_data::T; ++i) {
    for (size_t j = 1; j < test_data::N - 1; ++j)
      u_next[j] = test_data::d2x[0] * u[j - 1] + test_data::d2x[1] * u[j] + test_data::d2x[2] * u[j + 1];
    std::swap(u, u_next);
  }
  for (size_t i = 0; i < test_data::T; ++i) {
    test_data::d2x.apply_to<double, std::vector<double>, std::vector<double>>(x_next, x, {test_data::N});

    // restore the fixed boundaries to match the manual loop
    // the manual loop does not touch indices 0 and N-1
    // overwrite the calculated boundary values with the previous ones
    x_next[0] = x[0];
    x_next[test_data::N - 1] = x[test_data::N - 1];

    std::swap(x, x_next);
  }

  for (size_t j = 0; j < test_data::N; ++j) {
    ASSERT_EQ(u_next[j], x_next[j]) << "Index " << j;
    ASSERT_EQ(u[j], x[j]) << "Index " << j;
  }
}
