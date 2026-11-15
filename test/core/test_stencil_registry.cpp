#include <gtest/gtest.h>

#include <ozaki/utils/stencil_registry.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

#include <quadmath.h>

static StencilSpec make_heat_stencil() {
  return {"Heat",
          {3, 3},
          {1, 1},
          {CoeffStorageType(0), CoeffStorageType(3.0 / 16), CoeffStorageType(0), CoeffStorageType(3.0 / 16),
           CoeffStorageType(4.0 / 16), CoeffStorageType(3.0 / 16), CoeffStorageType(0), CoeffStorageType(3.0 / 16),
           CoeffStorageType(0)},
          OriginalPrecision::Double};
}

static std::vector<StencilSpec> make_test_stencils() {
  return {
      make_heat_stencil(),
      {"AdvectionDiffusion",
       {3, 3},
       {1, 1},
       {CoeffStorageType(0), CoeffStorageType(11.0 / 61), CoeffStorageType(0), CoeffStorageType(11.0 / 61),
        CoeffStorageType(15.0 / 61), CoeffStorageType(12.0 / 61), CoeffStorageType(0), CoeffStorageType(12.0 / 61),
        CoeffStorageType(0)},
       OriginalPrecision::Double},
      {"BoxAvg",
       {3, 3},
       {1, 1},
       {CoeffStorageType(1.0 / 9), CoeffStorageType(1.0 / 9), CoeffStorageType(1.0 / 9), CoeffStorageType(1.0 / 9),
        CoeffStorageType(1.0 / 9), CoeffStorageType(1.0 / 9), CoeffStorageType(1.0 / 9), CoeffStorageType(1.0 / 9),
        CoeffStorageType(1.0 / 9)},
       OriginalPrecision::Double},
      {"BoxRandom",
       {3, 3},
       {1, 1},
       {CoeffStorageType(-3.0 / 34), CoeffStorageType(2.0 / 34), CoeffStorageType(1.0 / 34),
        CoeffStorageType(-8.0 / 34), CoeffStorageType(4.0 / 34), CoeffStorageType(-5.0 / 34),
        CoeffStorageType(-1.0 / 34), CoeffStorageType(-1.0 / 34), CoeffStorageType(9.0 / 34)},
       OriginalPrecision::Double},
  };
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
      {{3, 3}, 9}, {{7, 7}, 49}, {{15, 15}, 225}, {{3, 3, 3}, 27}, {{5, 5, 5}, 125},
  };

  for (const auto &tc : cases) {
    StencilSpec spec;
    spec.name = "test";
    spec.shape = tc.shape;
    for (int d : tc.shape)
      spec.offsets.push_back(d / 2);
    spec.coeffs.resize(tc.expected_size, CoeffStorageType(1));

    EXPECT_NO_THROW(ozaki::make_stencil(spec))
        << "Failed for shape with " << tc.shape.size() << " dims, first dim = " << tc.shape[0];
  }
}

TEST(MakeStencil, UnsupportedShape_Throws) {
  StencilSpec spec;
  spec.name = "test";
  spec.shape = {5, 5};
  spec.offsets = {2, 2};
  spec.coeffs.resize(25, CoeffStorageType(1));

  EXPECT_THROW(ozaki::make_stencil(spec), std::invalid_argument);
}

TEST(MakeStencil, CoeffPreservation) {
  StencilSpec spec;
  spec.name = "test";
  spec.shape = {3, 3};
  spec.offsets = {1, 1};
  spec.coeffs = {CoeffStorageType(1), CoeffStorageType(-2), CoeffStorageType(1),
                 CoeffStorageType(0), CoeffStorageType(3),  CoeffStorageType(0),
                 CoeffStorageType(1), CoeffStorageType(-2), CoeffStorageType(1)};

  auto variant = ozaki::make_stencil(spec);

  // Verify coefficients survive the variant round-trip
  std::visit(
      [&](const auto &stencil) {
        ASSERT_EQ(stencil.size, 9u);
        EXPECT_EQ(static_cast<double>(stencil.coeffs[0]), 1.0);
        EXPECT_EQ(static_cast<double>(stencil.coeffs[1]), -2.0);
        EXPECT_EQ(static_cast<double>(stencil.coeffs[2]), 1.0);
      },
      variant);
}
