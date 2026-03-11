#include <gtest/gtest.h>

#include <ozaki/utils/stencil_registry.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

#ifdef __GNUC__
#include <quadmath.h>
#endif

// ============================================================
// find_integer_basis tests
// ============================================================

TEST(FindIntegerBasis, AllZero) {
  StencilSpec spec;
  spec.coeffs = {CoeffStorageType(0), CoeffStorageType(0), CoeffStorageType(0)};
  auto a = ozi::find_integer_basis(spec);
  EXPECT_EQ(a, CoeffStorageType(1));
}

TEST(FindIntegerBasis, SingleNonZero) {
  StencilSpec spec;
  spec.coeffs = {CoeffStorageType(0), CoeffStorageType(5), CoeffStorageType(0)};
  auto a = ozi::find_integer_basis(spec);
  // a should divide 5 evenly; smallest a = 5 (since abs_min=5, k=1 -> a=5, and 5/5=1 integer)
  EXPECT_GT(a, CoeffStorageType(0));
  // All coeffs / a should be integers
  for (const auto &c : spec.coeffs) {
    if (c != CoeffStorageType(0)) {
      CoeffStorageType ratio = c / a;
      CoeffStorageType rounded = CoeffStorageType(static_cast<int64_t>(
          static_cast<double>(ratio) + (static_cast<double>(ratio) >= 0 ? 0.5 : -0.5)));
      CoeffStorageType diff = ratio - rounded;
      if (diff < CoeffStorageType(0)) diff = -diff;
      EXPECT_LT(static_cast<double>(diff), 1e-9);
    }
  }
}

TEST(FindIntegerBasis, RationalCoeffs_Heat) {
  // Heat stencil: coefficients are multiples of 1/16
  StencilSpec spec;
  spec.coeffs = {CoeffStorageType(0), CoeffStorageType(3.0/16), CoeffStorageType(0),
                 CoeffStorageType(3.0/16), CoeffStorageType(4.0/16), CoeffStorageType(3.0/16),
                 CoeffStorageType(0), CoeffStorageType(3.0/16), CoeffStorageType(0)};
  auto a = ozi::find_integer_basis(spec);
  EXPECT_GT(a, CoeffStorageType(0));
  // All non-zero coefficients / a should be integers
  for (const auto &c : spec.coeffs) {
    if (c != CoeffStorageType(0)) {
      CoeffStorageType ratio = c / a;
      CoeffStorageType rounded = CoeffStorageType(static_cast<int64_t>(
          static_cast<double>(ratio) + (static_cast<double>(ratio) >= 0 ? 0.5 : -0.5)));
      CoeffStorageType diff = ratio - rounded;
      if (diff < CoeffStorageType(0)) diff = -diff;
      EXPECT_LT(static_cast<double>(diff), 1e-9);
    }
  }
}

TEST(FindIntegerBasis, NoExactBasis) {
  // Irrational coefficients: pi, e, sqrt(2)
  StencilSpec spec;
  spec.coeffs = {CoeffStorageType(M_PI), CoeffStorageType(M_E), CoeffStorageType(M_SQRT2)};
  auto a = ozi::find_integer_basis(spec);
  // Should return 0 sentinel (no exact integer basis found)
  EXPECT_EQ(a, CoeffStorageType(0));
}

TEST(FindIntegerBasis, LargeIntegerCoeffs) {
  StencilSpec spec;
  spec.coeffs = {CoeffStorageType(100), CoeffStorageType(-511), CoeffStorageType(100)};
  auto a = ozi::find_integer_basis(spec);
  EXPECT_GT(a, CoeffStorageType(0));
  // All coefficients / a should be integers
  for (const auto &c : spec.coeffs) {
    CoeffStorageType ratio = c / a;
    CoeffStorageType rounded = CoeffStorageType(static_cast<int64_t>(
        static_cast<double>(ratio) + (static_cast<double>(ratio) >= 0 ? 0.5 : -0.5)));
    CoeffStorageType diff = ratio - rounded;
    if (diff < CoeffStorageType(0)) diff = -diff;
    EXPECT_LT(static_cast<double>(diff), 1e-9);
  }
}

// ============================================================
// compute_m_needed tests
// ============================================================

TEST(ComputeMNeeded, SmallIntegers) {
  StencilSpec spec;
  spec.coeffs = {CoeffStorageType(1), CoeffStorageType(-2), CoeffStorageType(1)};
  int m = ozi::compute_m_needed(spec, CoeffStorageType(1));
  EXPECT_EQ(m, 1);
}

TEST(ComputeMNeeded, FourthOrderFD) {
  StencilSpec spec;
  spec.coeffs = {CoeffStorageType(-1), CoeffStorageType(16), CoeffStorageType(-30),
                 CoeffStorageType(16), CoeffStorageType(-1)};
  int m = ozi::compute_m_needed(spec, CoeffStorageType(1));
  EXPECT_EQ(m, 1);
}

TEST(ComputeMNeeded, TwoSlice) {
  StencilSpec spec;
  spec.coeffs = {CoeffStorageType(100), CoeffStorageType(-511), CoeffStorageType(100)};
  int m = ozi::compute_m_needed(spec, CoeffStorageType(1));
  EXPECT_EQ(m, 2);
}

TEST(ComputeMNeeded, ThreeSlice) {
  // Coprime coefficients with max=32769 (16 bits), p_max=15, M=3
  StencilSpec spec;
  spec.coeffs = {CoeffStorageType(-1), CoeffStorageType(16385), CoeffStorageType(-32769),
                 CoeffStorageType(16385), CoeffStorageType(-1)};
  int m = ozi::compute_m_needed(spec, CoeffStorageType(1));
  EXPECT_EQ(m, 3);
}

TEST(ComputeMNeeded, AllZero) {
  StencilSpec spec;
  spec.coeffs = {CoeffStorageType(0), CoeffStorageType(0), CoeffStorageType(0)};
  // a=1 (sentinel from find_integer_basis for all-zero)
  int m = ozi::compute_m_needed(spec, CoeffStorageType(1));
  EXPECT_EQ(m, 1);
}

// ============================================================
// compute_stencil_params tests
// ============================================================

TEST(ComputeStencilParams, SimpleD2x) {
  StencilSpec spec;
  spec.name = "d2x";
  spec.shape = {3};
  spec.offsets = {1};
  spec.coeffs = {CoeffStorageType(1), CoeffStorageType(-2), CoeffStorageType(1)};

  auto p = ozi::compute_stencil_params(spec);
  EXPECT_EQ(p.a_coeff, CoeffStorageType(1));
  EXPECT_EQ(p.m, 1);
  EXPECT_TRUE(p.exact_basis);
  EXPECT_EQ(p.extra_msb_shift, 0);
}

TEST(ComputeStencilParams, HeatStencil) {
  const auto &stencils = ozi::default_stencils();
  ASSERT_GT(stencils.size(), 0u);
  const auto &heat = stencils[0]; // "Heat"
  EXPECT_EQ(heat.name, "Heat");

  auto p = ozi::compute_stencil_params(heat);
  EXPECT_TRUE(p.exact_basis);
  EXPECT_EQ(p.m, 1);
  // Heat stencil: a=1/16, effective_a=1/16 * 2^0 = 1/16, scaling_factor = 16
  EXPECT_GT(static_cast<double>(p.scaling_factor), 1.0);
}

TEST(ComputeStencilParams, LargeL1) {
  StencilSpec spec;
  spec.name = "large";
  spec.shape = {3};
  spec.offsets = {1};
  spec.coeffs = {CoeffStorageType(100), CoeffStorageType(-511), CoeffStorageType(100)};

  auto p = ozi::compute_stencil_params(spec);
  EXPECT_GT(p.extra_msb_shift, 0);
  EXPECT_EQ(p.m, 2);
}

TEST(ComputeStencilParams, IrrationalFallback) {
  StencilSpec spec;
  spec.name = "irrational";
  spec.shape = {3};
  spec.offsets = {1};
  // Use truly incommensurable coefficients (pi, e, sqrt(2))
  spec.coeffs = {CoeffStorageType(M_PI), CoeffStorageType(M_E), CoeffStorageType(M_SQRT2)};

  auto p = ozi::compute_stencil_params(spec);
  EXPECT_FALSE(p.exact_basis);
  EXPECT_GT(static_cast<double>(p.a_coeff), 0.0);
}

// ============================================================
// make_stencil tests
// ============================================================

TEST(MakeStencil, AllSupportedShapes) {
  struct ShapeCase {
    std::vector<int> shape;
    size_t expected_size;
  };
  std::vector<ShapeCase> cases = {
      {{3}, 3},     {{5}, 5},     {{7}, 7},
      {{3, 3}, 9},  {{5, 5}, 25}, {{7, 7}, 49},
      {{3, 3, 3}, 27}, {{5, 5, 5}, 125},
  };

  for (const auto &tc : cases) {
    StencilSpec spec;
    spec.name = "test";
    spec.shape = tc.shape;
    for (int d : tc.shape)
      spec.offsets.push_back(d / 2);
    spec.coeffs.resize(tc.expected_size, CoeffStorageType(1));

    EXPECT_NO_THROW(ozi::make_stencil(spec))
        << "Failed for shape with " << tc.shape.size() << " dims, first dim = " << tc.shape[0];
  }
}

TEST(MakeStencil, UnsupportedShape_Throws) {
  StencilSpec spec;
  spec.name = "test";
  spec.shape = {9};
  spec.offsets = {4};
  spec.coeffs.resize(9, CoeffStorageType(1));

  EXPECT_THROW(ozi::make_stencil(spec), std::invalid_argument);
}

TEST(MakeStencil, CoeffPreservation) {
  StencilSpec spec;
  spec.name = "test";
  spec.shape = {3};
  spec.offsets = {1};
  spec.coeffs = {CoeffStorageType(1), CoeffStorageType(-2), CoeffStorageType(1)};

  auto variant = ozi::make_stencil(spec);

  // Verify coefficients survive the variant round-trip
  std::visit([&](const auto &stencil) {
    ASSERT_EQ(stencil.size, 3u);
    EXPECT_EQ(static_cast<double>(stencil.coeffs[0]), 1.0);
    EXPECT_EQ(static_cast<double>(stencil.coeffs[1]), -2.0);
    EXPECT_EQ(static_cast<double>(stencil.coeffs[2]), 1.0);
  }, variant);
}

// ============================================================
// default_stencils tests
// ============================================================

TEST(DefaultStencils, Count) {
  const auto &stencils = ozi::default_stencils();
  EXPECT_EQ(stencils.size(), 4u);
}

TEST(DefaultStencils, Names) {
  const auto &stencils = ozi::default_stencils();
  ASSERT_EQ(stencils.size(), 4u);
  EXPECT_EQ(stencils[0].name, "Heat");
  EXPECT_EQ(stencils[1].name, "AdvectionDiffusion");
  EXPECT_EQ(stencils[2].name, "BoxAvg");
  EXPECT_EQ(stencils[3].name, "BoxRandom");
}

TEST(DefaultStencils, AllValid) {
  const auto &stencils = ozi::default_stencils();
  for (const auto &s : stencils) {
    EXPECT_EQ(s.shape.size(), 2u) << "Stencil " << s.name << " should be 2D";
    EXPECT_EQ(s.shape[0], 3) << "Stencil " << s.name << " first dim should be 3";
    EXPECT_EQ(s.shape[1], 3) << "Stencil " << s.name << " second dim should be 3";
    EXPECT_EQ(s.coeffs.size(), 9u) << "Stencil " << s.name << " should have 9 coefficients";
  }
}
