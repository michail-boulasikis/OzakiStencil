#include <gtest/gtest.h>

#include <ozaki/core/stencil.hpp>
#include <ozaki/cpu/ozi.hpp>

#include <bit>
#include <climits>
#include <cstdint>
#include <random>
#include <vector>

// ---- independent helpers that mirror the implementation logic ---------------

// Returns the "p_max" the implementation would compute for a given set of
// integer coefficients (position, 0-indexed from LSB, of the highest set bit
// across all absolute values; -1 when every value is zero).
static int32_t expected_p_max(const std::vector<int64_t> &int_coeffs) {
  int32_t range_msb = 64;
  for (int64_t v : int_coeffs) {
    uint64_t abs_val = (v < 0) ? static_cast<uint64_t>(-v) : static_cast<uint64_t>(v);
    if (int32_t msb = std::countl_zero(abs_val); msb < range_msb)
      range_msb = msb;
  }
  return 63 - range_msb;
}

// Returns the shift used for the k-th slice.
// k=0 is clamped to non-negative (matching slice_stencil); k>0 can be negative.
static int32_t expected_shift(int32_t p_max, int32_t target_bits, int32_t k) {
  int32_t shift = p_max - (target_bits - 1) - target_bits * k;
  if (k == 0)
    shift = std::max(0, shift);
  return shift;
}

// Verifies that Sum_k  slice_k[i] * 2^shift_k  ==  int_coeff[i]  for every i.
// This is the core correctness invariant of slice_stencil.
template <int32_t N, typename IntStencilType>
void assert_reconstruction(const ozi::StencilSlices<IntStencilType, N> &result,
                            const std::vector<int64_t> &int_coeffs, int32_t p_max) {
  using Target = typename IntStencilType::value_type;
  constexpr int32_t target_bits = static_cast<int32_t>(CHAR_BIT * sizeof(Target)) - 1;

  for (size_t i = 0; i < IntStencilType::size; ++i) {
    int64_t reconstructed = 0;
    for (int32_t k = 0; k < N; ++k) {
      int32_t shift = expected_shift(p_max, target_bits, k);
      if (shift >= 0)
        reconstructed += static_cast<int64_t>(result.slices[k].coeffs[i]) << shift;
      else
        reconstructed += static_cast<int64_t>(result.slices[k].coeffs[i]) >> (-shift);
    }
    EXPECT_EQ(reconstructed, int_coeffs[i]) << "Reconstruction mismatch at coefficient index " << i;
  }
}

template <typename T> class SliceStencilTest : public ::testing::Test {
protected:
  std::vector<T> generate_random(size_t n, unsigned seed = 0x42) {
    std::mt19937 eng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<T> v(n);
    for (auto &x : v)
      x = static_cast<T>(dist(eng));
    return v;
  }
};

#ifdef OZAKI_HAVE_FLOAT128
using TestTypes = ::testing::Types<float, double, long double, __float128>;
#else
using TestTypes = ::testing::Types<float, double, long double>;
#endif

TYPED_TEST_SUITE(SliceStencilTest, TestTypes);

// ---- Test 1: All-zero coefficients (edge case) ------------------------------
TYPED_TEST(SliceStencilTest, AllZeroCoefficients) {
  using Scalar = TypeParam;
  using Traits = ozicpu::OzakiFP_Int8<Scalar>;

  ozi::Stencil<Scalar, 3> s; // default-constructed: all coeffs are zero

  auto slices = ozicpu::slice_stencil<Traits>(s, static_cast<Scalar>(1));

  for (size_t i = 0; i < 3; ++i)
    EXPECT_EQ(slices.slices[0].coeffs[i], int8_t(0));
}

// ---- Test 2: Offset preservation -------------------------------------------
// The offsets of the original stencil must be forwarded to every slice.
TYPED_TEST(SliceStencilTest, OffsetPreservation) {
  using Scalar = TypeParam;
  using Traits = ozicpu::OzakiFP_Int8<Scalar>;

  // Use a non-default offset to make the check meaningful.
  ozi::Stencil<Scalar, 3> s(
      {static_cast<Scalar>(1), static_cast<Scalar>(-2), static_cast<Scalar>(1)},
      {0} // offset = 0, not the default (size/2 = 1)
  );

  auto slices = ozicpu::slice_stencil<Traits>(s, static_cast<Scalar>(1));
  EXPECT_EQ(slices.slices[0].offsets, s.offsets);
}

// ---- Test 3: 2nd-order central FD -- exact N=1 slice ------------------------
// Stencil {1, -2, 1} (a = 1): int_coeffs = {1, -2, 1}.
// p_max = 1 (MSB of |2|), shift_0 = max(0, 1-6) = 0 -> slice_0 = {1, -2, 1}.
TYPED_TEST(SliceStencilTest, SecondOrderFDSingleSlice) {
  using Scalar = TypeParam;
  using Traits = ozicpu::OzakiFP_Int8<Scalar>;

  ozi::Stencil<Scalar, 3> s(
      {static_cast<Scalar>(1), static_cast<Scalar>(-2), static_cast<Scalar>(1)});
  Scalar a = static_cast<Scalar>(1);

  auto slices = ozicpu::slice_stencil<Traits>(s, a);

  EXPECT_EQ(slices.slices[0].coeffs[0], int8_t( 1));
  EXPECT_EQ(slices.slices[0].coeffs[1], int8_t(-2));
  EXPECT_EQ(slices.slices[0].coeffs[2], int8_t( 1));
  EXPECT_EQ(slices.slices[0].offsets, s.offsets);
}

// ---- Test 4: 4th-order central FD -- exact N=1 slice ------------------------
// Stencil {-1, 16, -30, 16, -1} (a = 1): int_coeffs fit in a single int8_t
// slice because |30| < 64 < 127 (MSB at bit 4, shift_0 = max(0, 4-6) = 0).
TYPED_TEST(SliceStencilTest, FourthOrderFDExactSingleSlice) {
  using Scalar = TypeParam;
  using Traits = ozicpu::OzakiFP_Int8<Scalar>;

  ozi::Stencil<Scalar, 5> s({
      static_cast<Scalar>(-1),
      static_cast<Scalar>(16),
      static_cast<Scalar>(-30),
      static_cast<Scalar>(16),
      static_cast<Scalar>(-1),
  });
  Scalar a = static_cast<Scalar>(1);

  auto slices = ozicpu::slice_stencil<Traits>(s, a);

  EXPECT_EQ(slices.slices[0].coeffs[0], int8_t( -1));
  EXPECT_EQ(slices.slices[0].coeffs[1], int8_t( 16));
  EXPECT_EQ(slices.slices[0].coeffs[2], int8_t(-30));
  EXPECT_EQ(slices.slices[0].coeffs[3], int8_t( 16));
  EXPECT_EQ(slices.slices[0].coeffs[4], int8_t( -1));
  EXPECT_EQ(slices.slices[0].offsets, s.offsets);
}

// ---- Test 5: Physical 4th-order FD with h^2 normalization -------------------
// The canonical finite-difference approximation to d^2u/dx^2 of order 4 is
//
//   (d2u/dx2)_i ~= (-u_{i-2} + 16 u_{i-1} - 30 u_i + 16 u_{i+1} - u_{i+2})
//                  / (12 h2)
//
// Written as a stencil, its coefficients are
//   c_k = {-1, 16, -30, 16, -1} / (12 h2),
// so the natural normalization factor is a = 1/(12 h2), which recovers the
// exact integers {-1, 16, -30, 16, -1}.
//
// We construct the stencil by multiplying the integers by a in __float128
// precision to guarantee that the float representation is as close as possible
// to the ideal, and we pass the same (cast) a to slice_stencil.
TYPED_TEST(SliceStencilTest, FourthOrderFDPhysicalNormalization) {
  using Scalar = TypeParam;
  using Traits = ozicpu::OzakiFP_Int8<Scalar>;

#ifdef OZAKI_HAVE_FLOAT128
  using HP = __float128;
#else
  using HP = long double;
#endif

  // Use h = 0.5 (exactly representable as a binary fraction) to minimise
  // rounding when forming the stencil coefficients.
  const HP hp_h  = static_cast<HP>(0.5);
  const HP hp_a  = static_cast<HP>(1) / (static_cast<HP>(12) * hp_h * hp_h);
  const Scalar a = static_cast<Scalar>(hp_a);

  const int64_t icoeffs[] = {-1, 16, -30, 16, -1};
  ozi::Stencil<Scalar, 5> s;
  s.offsets = {2}; // default offset for size-5 stencil
  for (size_t i = 0; i < 5; ++i)
    s.coeffs[i] = static_cast<Scalar>(static_cast<HP>(icoeffs[i]) * hp_a);

  auto slices = ozicpu::slice_stencil<Traits>(s, a);

  // HP-precision division inside slice_stencil recovers the exact integers.
  for (size_t i = 0; i < 5; ++i)
    EXPECT_EQ(static_cast<int64_t>(slices.slices[0].coeffs[i]), icoeffs[i])
        << "Mismatch at coefficient index " << i;
}

// ---- Test 6: N=2 -- coefficients that need two slices -----------------------
// Stencil {100, -511, 100} (a = 1):
//   |511| = 0x1FF (9 bits) -> p_max = 8, shift_0 = max(0, 8-6) = 2
//   Slice 0:  100>>2 = 25,  (-511)>>2 = -128 -> clamped -127,  100>>2 = 25
//             residuals:      0,                -511+508 = -3,         0
//   shift_1 = 8-6-7 = -5  (left-shift residual by 5)
//   Slice 1:   0, -3<<5 = -96, 0
//   Reconstruction: 25*4+0 = 100, -127*4+(-96>>5) = -127*4-3 = -511, 25*4+0 = 100
TYPED_TEST(SliceStencilTest, TwoSliceCoefficients) {
  using Scalar = TypeParam;
  using Traits = ozicpu::OzakiFP_Int8<Scalar>;

  ozi::Stencil<Scalar, 3> s({
      static_cast<Scalar>(100),
      static_cast<Scalar>(-511),
      static_cast<Scalar>(100),
  });
  Scalar a = static_cast<Scalar>(1);

  auto slices =
      ozicpu::slice_stencil<Traits, ozi::Stencil<Scalar, 3>, 2>(s, a);

  // Most-significant slice (shift = 2)
  EXPECT_EQ(slices.slices[0].coeffs[0], int8_t(  25));
  EXPECT_EQ(slices.slices[0].coeffs[1], int8_t(-127));
  EXPECT_EQ(slices.slices[0].coeffs[2], int8_t(  25));

  // Least-significant slice (shift = -5, i.e. left-shift by 5)
  EXPECT_EQ(slices.slices[1].coeffs[0], int8_t(  0));
  EXPECT_EQ(slices.slices[1].coeffs[1], int8_t(-96));
  EXPECT_EQ(slices.slices[1].coeffs[2], int8_t(  0));

  // Offsets must be preserved in both slices.
  EXPECT_EQ(slices.slices[0].offsets, s.offsets);
  EXPECT_EQ(slices.slices[1].offsets, s.offsets);
}

// ---- Test 7: Reconstruction invariant for N=2 ------------------------------
// For any stencil and N > 1, the bit-sum of all slices must equal the
// original integer coefficient vector.
TYPED_TEST(SliceStencilTest, TwoSliceReconstructionInvariant) {
  using Scalar = TypeParam;
  using Traits = ozicpu::OzakiFP_Int8<Scalar>;

  // Use a stencil with a wide dynamic range: the 4th-order biharmonic weights
  // scaled up to require two slices. 300 > 127, so it overflows int8_t.
  ozi::Stencil<Scalar, 5> s({
      static_cast<Scalar>( 300),
      static_cast<Scalar>(-1200),
      static_cast<Scalar>( 1800),
      static_cast<Scalar>(-1200),
      static_cast<Scalar>( 300),
  });
  Scalar a = static_cast<Scalar>(1);

  auto slices =
      ozicpu::slice_stencil<Traits, ozi::Stencil<Scalar, 5>, 2>(s, a);

  const std::vector<int64_t> int_coeffs = {300, -1200, 1800, -1200, 300};
  const int32_t p_max = expected_p_max(int_coeffs);
  assert_reconstruction(slices, int_coeffs, p_max);
}

// ---- Test 8: Reconstruction invariant for N=3 ------------------------------
// With three slices we can capture 21 bits of integer coefficient range.
TYPED_TEST(SliceStencilTest, ThreeSliceReconstructionInvariant) {
  using Scalar = TypeParam;
  using Traits = ozicpu::OzakiFP_Int8<Scalar>;

  // Coefficients whose integers span > 14 bits so a third slice is needed.
  const std::vector<int64_t> int_coeffs = {-100000, 400000, -600000, 400000, -100000};
  ozi::Stencil<Scalar, 5> s;
  for (size_t i = 0; i < 5; ++i)
    s.coeffs[i] = static_cast<Scalar>(int_coeffs[i]);
  Scalar a = static_cast<Scalar>(1);

  auto slices =
      ozicpu::slice_stencil<Traits, ozi::Stencil<Scalar, 5>, 3>(s, a);

  const int32_t p_max = expected_p_max(int_coeffs);
  assert_reconstruction(slices, int_coeffs, p_max);
}

// ---- Test 9: Application accuracy for N=1 ----------------------------------
// Applying the reconstructed float stencil (a * int8_coeff) to random data
// must produce the same result as the original float stencil when the integer
// coefficients are exact (a = 1, integer-valued coefficients).
TYPED_TEST(SliceStencilTest, ApplicationAccuracyN1) {
  using Scalar = TypeParam;
  using Traits = ozicpu::OzakiFP_Int8<Scalar>;

  ozi::Stencil<Scalar, 5> s_orig({
      static_cast<Scalar>(-1),
      static_cast<Scalar>(16),
      static_cast<Scalar>(-30),
      static_cast<Scalar>(16),
      static_cast<Scalar>(-1),
  });
  Scalar a = static_cast<Scalar>(1);

  auto slices = ozicpu::slice_stencil<Traits>(s_orig, a);

  // Build a float stencil from the int8_t slice (shift = 0, a = 1 here).
  ozi::Stencil<Scalar, 5> s_recon;
  s_recon.offsets = slices.slices[0].offsets;
  for (size_t i = 0; i < 5; ++i)
    s_recon.coeffs[i] = static_cast<Scalar>(slices.slices[0].coeffs[i]) * a;

  // Apply both stencils to random input.
  constexpr size_t N = 256;
  auto data = this->generate_random(N);

  std::vector<Scalar> result_orig(N, static_cast<Scalar>(0));
  std::vector<Scalar> result_recon(N, static_cast<Scalar>(0));
  s_orig.template apply_to<Scalar>(result_orig, data, {N});
  s_recon.template apply_to<Scalar>(result_recon, data, {N});

  // For integer coefficients with a=1 the two reconstructed stencils are
  // bit-identical, so the results must match exactly.
  for (size_t i = 2; i < N - 2; ++i)
    ASSERT_EQ(result_orig[i], result_recon[i]) << "Mismatch at index " << i;
}

// ---- Non-typed test: physical normalization with double --------------------
// This is the most physically relevant scenario: the 4th-order second-derivative
// FD operator on a grid with spacing h, applied to u(x) = x2.
// The exact answer is 2.0 everywhere, so we can check absolute error.
//
// The key subtlety: passing `a` as __float128 forces HighPrec = __float128
// inside slice_stencil, so the division coeff[i] / a is performed with quad
// precision and returns the exact integers {-1, 16, -30, 16, -1} even though
// the stencil coefficients themselves are only double-precision.
TEST(SliceStencilPhysical, FourthOrderOnXSquared) {
  using Scalar = double;
  using Traits = ozicpu::OzakiFP_Int8<Scalar>;

  constexpr size_t N = 32;
  const Scalar h = 0.1;

#ifdef OZAKI_HAVE_FLOAT128
  // Compute a in __float128 so HighPrec is deduced as __float128 in
  // slice_stencil, guaranteeing exact integer recovery regardless of the
  // double rounding of 1/(12 h2).
  using HP = __float128;
#else
  using HP = long double;
#endif
  const HP hp_a = static_cast<HP>(1) /
                  (static_cast<HP>(12) * static_cast<HP>(h) * static_cast<HP>(h));
  const Scalar a = static_cast<Scalar>(hp_a);

  // Build the stencil coefficients as integer * a (double precision).
  const int64_t icoeffs[] = {-1, 16, -30, 16, -1};
  ozi::Stencil<Scalar, 5> s;
  s.offsets = {2};
  for (size_t i = 0; i < 5; ++i)
    s.coeffs[i] = static_cast<Scalar>(static_cast<HP>(icoeffs[i]) * hp_a);

  // Pass hp_a so HighPrec = HP; the division is then exact.
  auto slices = ozicpu::slice_stencil<Traits>(s, hp_a);

  // Verify that the HP-precision division recovers the exact integers.
  for (size_t i = 0; i < 5; ++i)
    EXPECT_EQ(static_cast<int64_t>(slices.slices[0].coeffs[i]), icoeffs[i])
        << "Integer recovery failed at coefficient index " << i;

  // Reconstruct the double stencil: coeff_recon[i] = a * int8_coeff[i].
  // Since int8_coeff[i] == icoeff[i] (verified above), this exactly reproduces
  // the original double-precision stencil s.
  ozi::Stencil<Scalar, 5> s_recon;
  s_recon.offsets = slices.slices[0].offsets;
  for (size_t i = 0; i < 5; ++i)
    s_recon.coeffs[i] = a * static_cast<Scalar>(slices.slices[0].coeffs[i]);

  // Apply to u(x) = (i h)2; d2u/dx2 = 2 exactly for any h.
  // The 4th-order stencil captures degree-2 polynomials exactly, so
  // s_recon applied to u returns 2.0 regardless of the specific value of h.
  std::vector<Scalar> u(N), result(N, 0.0);
  for (size_t i = 0; i < N; ++i)
    u[i] = static_cast<Scalar>(i) * h * static_cast<Scalar>(i) * h;

  s_recon.template apply_to<Scalar>(result, u, {N});

  for (size_t i = 2; i < N - 2; ++i)
    EXPECT_NEAR(result[i], 2.0, 1e-10) << "Large error at index " << i;
}
