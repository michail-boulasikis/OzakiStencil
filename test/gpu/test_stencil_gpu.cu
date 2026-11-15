#include <gtest/gtest.h>

#include <ozaki/gpu/stencil_integrator.cuh>
#include <ozaki/utils/stencil_registry.hpp>
#include <ozaki/utils/stencil_spec.hpp>
#include <ozaki/utils/zero_boundary.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool sm90_available() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    return false;
  }
  int major = 0;
  return cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) == cudaSuccess && major >= 9;
}

// A centered 3x3 stencil with the given (exactly representable) coefficients.
StencilSpec make_3x3(std::array<double, 9> coefficients) {
  StencilSpec spec;
  spec.name = "test3x3";
  spec.shape = {3, 3};
  spec.offsets = {1, 1};
  spec.specified_precision = OriginalPrecision::Double;
  spec.coeffs.assign(coefficients.begin(), coefficients.end());
  return spec;
}

// A stencil with exactly `active_slices` base-128 coefficient digits.
CoeffStorageType pow2(int32_t exponent) {
  return core::RealTraits<CoeffStorageType>::scalbn(static_cast<CoeffStorageType>(1), exponent);
}

StencilSpec make_capacity_stencil(int32_t dimensions, int32_t side, int32_t active_slices) {
  StencilSpec spec;
  spec.name = std::to_string(side) + "d" + std::to_string(dimensions) + "s" + std::to_string(active_slices);
  spec.shape.assign(static_cast<size_t>(dimensions), side);
  spec.offsets.assign(static_cast<size_t>(dimensions), side / 2);
  spec.specified_precision = OriginalPrecision::Float128;
  size_t coefficient_count = 1;
  for (int32_t dimension = 0; dimension < dimensions; ++dimension)
    coefficient_count *= static_cast<size_t>(side);
  spec.coeffs.resize(coefficient_count);
  for (size_t coefficient = 0; coefficient < coefficient_count; ++coefficient) {
    if (coefficient % 5 == 0)
      continue;
    const int32_t digit = static_cast<int32_t>(coefficient % 3) + 1;
    spec.coeffs[coefficient] =
        pow2(-core::SLICE_STEP) * static_cast<CoeffStorageType>((coefficient & 1U) == 0U ? digit : -digit);
  }
  const size_t marker = coefficient_count / 2;
  spec.coeffs[marker] = pow2(-core::SLICE_STEP);
  if (active_slices > 1)
    spec.coeffs[marker] += pow2(-core::SLICE_STEP * active_slices);
  return spec;
}


template <typename Real>
std::vector<Real> cpu_reference(const StencilSpec &spec, const Tensor<Real> &input, int32_t timesteps) {
  return std::visit(
      [&](const auto &source_stencil) -> std::vector<Real> {
        using SourceStencil = std::decay_t<decltype(source_stencil)>;
        using RealStencil = ozaki::RebindStencil_t<SourceStencil, Real>;
        RealStencil stencil;
        stencil.offsets = source_stencil.offsets;
        for (size_t i = 0; i < source_stencil.coeffs.size(); ++i)
          stencil.coeffs[i] = static_cast<Real>(source_stencil.coeffs[i]);
        typename RealStencil::shape_type shape{};
        std::copy(input.shape.begin(), input.shape.end(), shape.begin());
        // Same homogeneous Dirichlet problem as StencilIntegrator: zero halo.
        const Tensor<Real> bounded = zero_boundary_copy(input, static_cast<size_t>(spec.shape.front() / 2));
        // apply_to writes only interior cells, so both halos stay zero.
        std::vector<Real> current = bounded.data;
        std::vector<Real> next = bounded.data;
        for (int32_t step = 0; step < timesteps; ++step) {
          stencil.template apply_to<Real>(next, current, shape);
          std::swap(current, next);
        }
        return current;
      },
      ozaki::make_stencil(spec));
}

template <typename Real>
void expect_close(const std::vector<Real> &actual, const std::vector<Real> &expected,
                  long double relative_tolerance = 1.0e-11L) {
  ASSERT_EQ(actual.size(), expected.size());
  long double worst_ratio = 0;
  long double worst_error = 0;
  size_t worst_index = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const long double a = static_cast<long double>(actual[i]);
    const long double e = static_cast<long double>(expected[i]);
    const long double tolerance = relative_tolerance * std::max(1.0L, std::abs(e));
    const long double error = std::abs(a - e);
    if (error / tolerance > worst_ratio) {
      worst_ratio = error / tolerance;
      worst_error = error;
      worst_index = i;
    }
  }
  EXPECT_LE(worst_ratio, 1.0L) << "worst error " << static_cast<double>(worst_error) << " at linear index "
                               << worst_index;
}

TEST(ZeroBoundaryHost, ClearsCompleteHalo) {
  for (const size_t halo : {1U, 2U, 3U, 7U}) {
    Tensor<double> input;
    input.shape = {2 * halo + 3, 2 * halo + 5};
    input.data.assign(input.shape[0] * input.shape[1], 7.0);
    const Tensor<double> bounded = zero_boundary_copy(input, halo);
    for (size_t y = 0; y < input.shape[0]; ++y) {
      for (size_t x = 0; x < input.shape[1]; ++x) {
        const bool boundary = y < halo || y + halo >= input.shape[0] || x < halo || x + halo >= input.shape[1];
        EXPECT_EQ(bounded.data[y * input.shape[1] + x], boundary ? 0.0 : 7.0);
      }
    }
  }
}

TEST(ZeroBoundaryHost, ClearsThreeDimensionalHaloAndRejectsEmptyInterior) {
  Tensor<double> input;
  input.shape = {7, 8, 9};
  input.data.assign(7 * 8 * 9, -3.0);
  const Tensor<double> bounded = zero_boundary_copy(input, 2);
  for (size_t z = 0; z < input.shape[0]; ++z) {
    for (size_t y = 0; y < input.shape[1]; ++y) {
      for (size_t x = 0; x < input.shape[2]; ++x) {
        const bool boundary =
            z < 2 || z + 2 >= input.shape[0] || y < 2 || y + 2 >= input.shape[1] || x < 2 || x + 2 >= input.shape[2];
        const size_t index = (z * input.shape[1] + y) * input.shape[2] + x;
        EXPECT_EQ(bounded.data[index], boundary ? 0.0 : -3.0);
      }
    }
  }
  input.shape = {4, 8, 9};
  input.data.resize(4 * 8 * 9);
  EXPECT_THROW(zero_boundary_copy(input, 2), std::invalid_argument);
}

TEST(StencilIntegratorGpu, IdentityKeepsInteriorAndZeroHalo) {
  if (!sm90_available()) {
    GTEST_SKIP() << "SM90 or newer is required";
  }
  StencilSpec spec = make_3x3({0, 0, 0, 0, 1, 0, 0, 0, 0});
  spec.name = "identity";
  using Real = ozaki::gpu::IntegrationReal;
  Tensor<Real> input;
  input.shape = {9, 37};
  input.data.resize(9 * 37);
  for (size_t i = 0; i < input.data.size(); ++i) {
    input.data[i] = static_cast<Real>(static_cast<int32_t>(i % 29) - 14) / static_cast<Real>(7);
  }
  ozaki::gpu::StencilIntegrator integrator(spec, input, 10);
  integrator.integrate(10);
  const std::vector<Real> result = integrator.reconstruct();
  for (size_t y = 0; y < input.shape[0]; ++y) {
    for (size_t x = 0; x < input.shape[1]; ++x) {
      const size_t index = y * input.shape[1] + x;
      if (y == 0 || y + 1 == input.shape[0] || x == 0 || x + 1 == input.shape[1]) {
        EXPECT_EQ(result[index], static_cast<Real>(0));
      } else {
        EXPECT_NEAR(static_cast<double>(result[index]), static_cast<double>(input.data[index]), 1.0e-12);
      }
    }
  }
}

TEST(StencilIntegratorGpu, WindowFollowsAttenuationAndAmplification) {
  if (!sm90_available()) {
    GTEST_SKIP() << "SM90 or newer is required";
  }
  using Real = ozaki::gpu::IntegrationReal;
  Tensor<Real> input;
  input.shape = {9, 37};
  input.data.assign(9 * 37, static_cast<Real>(64));

  StencilSpec attenuation = make_3x3({0, 0, 0, 0, 0.0078125, 0, 0, 0, 0});
  attenuation.name = "attenuation_moves_window_down";
  ozaki::gpu::StencilIntegrator lowered(attenuation, input, 12);
  lowered.integrate(8);
  expect_close(lowered.reconstruct(), cpu_reference(attenuation, input, 8), 1.0e-12L);

  StencilSpec amplification = make_3x3({0, 0, 0, 0, 128, 0, 0, 0, 0});
  amplification.name = "amplification_moves_window_up";
  ozaki::gpu::StencilIntegrator raised(amplification, input, 12);
  raised.integrate(3);
  expect_close(raised.reconstruct(), cpu_reference(amplification, input, 3), 1.0e-12L);
}

// Each step grows the values by up to 9 * 127^2, i.e. by more than two base-128
// digits; the digit window must follow.
TEST(StencilIntegratorGpu, LargeGrowthPerStepMatchesCpuReference) {
  if (!sm90_available()) {
    GTEST_SKIP() << "SM90 or newer is required";
  }
  StencilSpec spec = make_3x3({127, 127, 127, 127, 127, 127, 127, 127, 127});
  spec.name = "large_l1";
  using Real = ozaki::gpu::IntegrationReal;
  Tensor<Real> input;
  input.shape = {7, 35};
  input.data.assign(7 * 35, static_cast<Real>(127));
  ozaki::gpu::StencilIntegrator integrator(spec, input, 12);
  integrator.integrate(3);
  expect_close(integrator.reconstruct(), cpu_reference(spec, input, 3), 1.0e-12L);
}

// With enough data slices, slicing and reconstructing (no step) is exact.
TEST(StencilIntegratorGpu, ZeroStepsReturnTheInputExactly) {
  if (!sm90_available()) {
    GTEST_SKIP() << "SM90 or newer is required";
  }
  using Real = ozaki::gpu::IntegrationReal;
  Tensor<Real> input;
  input.shape = {9, 37};
  input.data.resize(9 * 37);
  for (size_t i = 0; i < input.data.size(); ++i) {
    input.data[i] = core::RealTraits<Real>::scalbn(static_cast<Real>(static_cast<int32_t>(i % 31) - 15) / 3, -20);
  }
  ozaki::gpu::StencilIntegrator integrator(make_3x3({0, 0, 0, 0, 1, 0, 0, 0, 0}), input, 31);
  EXPECT_EQ(integrator.reconstruct(), zero_boundary_copy(input, 1).data);
}

struct KernelCase {
  int32_t dimensions;
  int32_t side;
  int32_t slices;
};

// Every kernel the correctness and performance pipelines use.
constexpr std::array kernel_cases{KernelCase{2, 3, 1},  KernelCase{2, 3, 2},  KernelCase{2, 3, 9},
                                  KernelCase{2, 7, 1},  KernelCase{2, 7, 2},  KernelCase{2, 7, 9},
                                  KernelCase{2, 15, 1}, KernelCase{2, 15, 2}, KernelCase{2, 15, 3},
                                  KernelCase{2, 15, 9}, KernelCase{3, 3, 1},  KernelCase{3, 3, 2},
                                  KernelCase{3, 3, 9},  KernelCase{3, 5, 1},  KernelCase{3, 5, 2},
                                  KernelCase{3, 5, 9}};

// Spans more than one CTA tile in every dimension with partial fringe tiles
// (2D tiles are 32 or 24 rows x 128 or 256 columns; 3D tiles 192 x 4 x 8 or 2).
std::vector<size_t> fringe_shape(const KernelCase &c) {
  const size_t halo = static_cast<size_t>(c.side / 2);
  if (c.dimensions == 2) {
    const size_t tile_cols = c.side == 15 ? 256 : 128;
    return {2 * 32 + 2 * halo + 3, tile_cols + 2 * halo + 5};
  }
  return {2 * 8 + 2 * halo + 3, 4 + 2 * halo + 3, 192 + 2 * halo + 5};
}

template <typename Real> Tensor<Real> make_input(const std::vector<size_t> &shape, bool dense) {
  Tensor<Real> input;
  input.shape = shape;
  size_t elements = 1;
  for (const size_t extent : shape)
    elements *= extent;
  input.data.resize(elements);
  for (size_t i = 0; i < elements; ++i) {
    // Small integers occupy only the top digit planes (sparse plane masks);
    // the dense pattern fills every plane.
    const Real value = static_cast<Real>(static_cast<int32_t>((7 * i + 3) % 17) - 8);
    input.data[i] = dense ? value / static_cast<Real>(3) : value;
  }
  return input;
}

TEST(StencilIntegratorGpu, EveryKernelMatchesCpuReference) {
  if (!sm90_available()) {
    GTEST_SKIP() << "SM90 or newer is required";
  }
  using Real = ozaki::gpu::IntegrationReal;
  for (const KernelCase c : kernel_cases) {
    const StencilSpec spec = make_capacity_stencil(c.dimensions, c.side, c.slices);
    for (const bool dense : {false, true}) {
      SCOPED_TRACE(spec.name + (dense ? " dense" : " sparse"));
      const Tensor<Real> input = make_input<Real>(fringe_shape(c), dense);
      try {
        ozaki::gpu::StencilIntegrator integrator(spec, input, 31);
        for (const int32_t timesteps : {1, 7}) {
          SCOPED_TRACE("timesteps=" + std::to_string(timesteps));
          integrator.reset();
          integrator.integrate(timesteps);
          expect_close(integrator.reconstruct(), cpu_reference(spec, input, timesteps));
        }
      } catch (const std::exception &error) {
        ADD_FAILURE() << "exception: " << error.what();
      }
    }
  }
}

TEST(StencilIntegratorGpu, FewDataSlicesStayAccurateForEveryKernel) {
  if (!sm90_available()) {
    GTEST_SKIP() << "SM90 or newer is required";
  }
  using Real = ozaki::gpu::IntegrationReal;
  for (const KernelCase c : kernel_cases) {
    const StencilSpec spec = make_capacity_stencil(c.dimensions, c.side, c.slices);
    const Tensor<Real> input = make_input<Real>(fringe_shape(c), true);
    const std::vector<Real> expected = cpu_reference(spec, input, 3);
    for (const int32_t data_slices : {2, 3, 5, 8}) {
      SCOPED_TRACE(spec.name + " data_slices=" + std::to_string(data_slices));
      ozaki::gpu::StencilIntegrator integrator(spec, input, data_slices);
      integrator.integrate(3);
      // A window of n base-128 digits resolves about 7 (n - 1) bits.
      expect_close(integrator.reconstruct(), expected, std::ldexp(1.0L, -7 * (data_slices - 1) + 8));
    }
  }
}

// Only 1, 2 and 9 coefficient slices (and 3 for 15x15) are supported; more
// than nine cannot be represented at all.
TEST(PackStencilHost, RejectsUnsupportedCoefficientSliceCounts) {
  for (const KernelCase c : {KernelCase{2, 7, 3}, KernelCase{2, 15, 4}, KernelCase{3, 5, 3}, KernelCase{2, 3, 8},
                             KernelCase{2, 7, 10}}) {
    SCOPED_TRACE(std::to_string(c.side) + "^" + std::to_string(c.dimensions) + " slices=" + std::to_string(c.slices));
    EXPECT_ANY_THROW(ozaki::gpu::pack_stencil(make_capacity_stencil(c.dimensions, c.side, c.slices)));
  }
}

} // namespace
