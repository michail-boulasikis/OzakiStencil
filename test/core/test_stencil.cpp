#include <gtest/gtest.h>
#include <ozaki/core/stencil.hpp>

#include <numeric>
#include <vector>

// The CPU stencil (used by the reference solver) must agree with a hand-written
// loop on the interior and leave the boundary untouched.
TEST(CpuStencil, ApplyToMatchesHandWrittenLoop) {
  constexpr size_t ny = 7, nx = 9;
  const ozaki::Stencil<double, 3, 3> laplace{{0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0}, {1, 1}};
  std::vector<double> input(ny * nx);
  std::iota(input.begin(), input.end(), 0.0);
  for (double &value : input) {
    value = value * value; // non-linear, so the Laplacian is non-zero
  }
  std::vector<double> output(ny * nx, -1.0);
  laplace.apply_to<double>(output, input, {ny, nx});

  for (size_t y = 0; y < ny; ++y) {
    for (size_t x = 0; x < nx; ++x) {
      const size_t i = y * nx + x;
      if (y == 0 || x == 0 || y + 1 == ny || x + 1 == nx) {
        EXPECT_EQ(output[i], -1.0) << y << "," << x;
      } else {
        const double expected = input[i - nx] + input[i - 1] - 4.0 * input[i] + input[i + 1] + input[i + nx];
        EXPECT_EQ(output[i], expected) << y << "," << x;
      }
    }
  }
}
