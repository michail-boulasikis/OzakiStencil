#ifndef HEADER_OZAKI_HPP
#define HEADER_OZAKI_HPP

#include "../core/stencil.hpp"
#include "../core/oztraits.hpp"

#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <concepts>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef __GNUC__
#define HighPrecType __float128
#else
#define HighPrecType long double
#endif


namespace ozicpu {
template <typename T> struct OzakiFP_Int8 {
static_assert(false, "Not implemented");
};

template <std::floating_point T>
  requires(sizeof(T) > 1)
struct OzakiFP_Int8<T> {
  using data_type = T;
  using target_type = int8_t;
  using acc_type = int32_t;
};

#ifdef __GNUC__
// Specialization for __float128 (not recognized by std::floating_point concept)
template <>
struct OzakiFP_Int8<__float128> {
  using data_type = __float128;
  using target_type = int8_t;
  using acc_type = int32_t;
};
#endif
} // namespace ozicpu

namespace ozi {

#ifdef __GNUC__
// CPU-side OzakiPairTraits for __float128
template <> struct OzakiPairTraits<ozicpu::OzakiFP_Int8<__float128>> {
  using Data = __float128;
  using Target = int8_t;
  using Acc = int32_t;

  static inline Target down_cast(Data a, int32_t scale) {
    constexpr auto LO = static_cast<int64_t>(std::numeric_limits<Target>::min() + 1);
    constexpr auto HI = static_cast<int64_t>(std::numeric_limits<Target>::max());
    __float128 scaled = OzakiTraits<Data>::scalbn(a, OzakiTraits<Target>::scaling_factor(scale));
    auto r = static_cast<int64_t>(llrintq(scaled));
    if (r < LO)
      r = LO;
    if (r > HI)
      r = HI;
    return static_cast<Target>(r);
  }

  static inline Data up_cast(Target a, int32_t scale) {
    return OzakiTraits<Data>::scalbn(static_cast<Data>(a), -OzakiTraits<Target>::scaling_factor(scale));
  }

  static inline Data up_cast_acc(Acc a, int32_t scale, int32_t n_mult) {
    return OzakiTraits<Data>::scalbn(static_cast<Data>(a), -(n_mult * (CHAR_BIT * sizeof(Target) - 1) - scale));
  }

  static void matmul(Acc *__restrict C, const Target *__restrict A, const Target *__restrict B, size_t m, size_t n, size_t k) {
    for (size_t mm = 0; mm < m; ++mm) {
      for (size_t nn = 0; nn < n; ++nn) {
        Acc res = 0;
        for (size_t kk = 0; kk < k; ++kk) {
          res += static_cast<Acc>(A[mm + m * kk]) * static_cast<Acc>(B[kk + nn * k]);
        }
        C[mm + m * nn] = res;
      }
    }
  }
};
#endif

template <std::floating_point T>
struct OzakiPairTraits<ozicpu::OzakiFP_Int8<T>> {
  using Data = typename ozicpu::OzakiFP_Int8<T>::data_type;
  using Target = typename ozicpu::OzakiFP_Int8<T>::target_type;
  using Acc = typename ozicpu::OzakiFP_Int8<T>::acc_type;

  static inline Target down_cast(Data a, int32_t scale) {
    constexpr auto LO = static_cast<int64_t>(std::numeric_limits<Target>::min() + 1);
    constexpr auto HI = static_cast<int64_t>(std::numeric_limits<Target>::max());
    auto r = llround(OzakiTraits<Data>::scalbn(a, OzakiTraits<Target>::scaling_factor(scale)));
    if (r < LO)
      r = LO;
    if (r > HI)
      r = HI;
    return static_cast<Target>(r);
  }

  static inline Data up_cast(Target a, int32_t scale) {
    return OzakiTraits<Data>::scalbn(static_cast<Data>(a), -OzakiTraits<Target>::scaling_factor(scale));
  }

  static inline Data up_cast_acc(Acc a, int32_t scale, int32_t n_mult) {
    return OzakiTraits<Data>::scalbn(static_cast<Data>(a), -(n_mult * (CHAR_BIT * sizeof(Target) - 1) - scale));
  }

  static void matmul(Acc *__restrict C, const Target *__restrict A, const Target *__restrict B, size_t m, size_t n, size_t k) {
    for (size_t mm = 0; mm < m; ++mm) {
      for (size_t nn = 0; nn < n; ++nn) {
        Acc res = 0;
        for (size_t kk = 0; kk < k; ++kk) {
          res += static_cast<Acc>(A[mm + m * kk]) * static_cast<Acc>(B[kk + nn * k]);
        }
        C[mm + m * nn] = res;
      }
    }
  }
};
} // namespace ozi

namespace ozicpu {
using namespace ozi;

template <OzakiPair O> struct OzakiSlices {
  std::vector<typename O::target_type> values;
  std::vector<int32_t> exponents;
  size_t width;

  std::span<const typename O::target_type> get_slice(size_t index) const {
    return {values.data() + (index * width), width};
  }
};

template <OzakiPair O> OzakiSlices<O> slice(std::span<const typename O::data_type> data, size_t max_slices = 0) {
  using Data = typename O::data_type;
  using Target = typename O::target_type;

  size_t n = data.size();
  OzakiSlices<O> result;
  result.width = n;

  size_t reserve_est = (max_slices > 0) ? max_slices : 2 * sizeof(Data);
  result.values.reserve(reserve_est * n);
  result.exponents.reserve(reserve_est);

  std::vector<Data> residual(data.begin(), data.end());

  size_t current_slice_idx = 0;

  while (true) {
    if (max_slices > 0 && current_slice_idx >= max_slices)
      break;

    Data abs_max = static_cast<Data>(0);
    for (const auto &val : residual)
      abs_max = OzakiTraits<Data>::max(OzakiTraits<Data>::abs(val), abs_max);

    if (!OzakiTraits<Data>::isfinite(abs_max))
      throw std::runtime_error("Ozaki scheme diverged: infinity detected.");
    if (abs_max == static_cast<Data>(0))
      break;

    int32_t exp;
    OzakiTraits<Data>::frexp(abs_max, &exp);
    result.exponents.push_back(exp);

    size_t start_offset = result.values.size();
    result.values.resize(start_offset + n);

    Target *current_slice_ptr = result.values.data() + start_offset;
    Data *residual_ptr = residual.data();

    for (size_t j = 0; j < n; ++j) {
      current_slice_ptr[j] = OzakiPairTraits<O>::down_cast(residual_ptr[j], exp);
      residual_ptr[j] -= OzakiPairTraits<O>::up_cast(current_slice_ptr[j], exp);
    }

    current_slice_idx++;
  }

  return result;
}

template <OzakiPair O, typename StencilType, int32_t N = 1, typename HighPrec = HighPrecType>
StencilSlices<RebindStencil_t<StencilType, typename O::target_type>, N>
slice_stencil(StencilType s, HighPrec a, int32_t extra_msb_shift = 0, int32_t n_emulated = 0) {
  using Target = typename O::target_type;
  using IntStencilType = RebindStencil_t<StencilType, Target>;

  static_assert(N >= 1, "N must be at least 1");

  // Number of usable bits per int8_t slice (excludes the disallowed INT8_MIN value).
  constexpr int32_t target_bits = static_cast<int32_t>(CHAR_BIT * sizeof(Target)) - 1;
  constexpr int64_t LO = static_cast<int64_t>(std::numeric_limits<Target>::min()) + 1;
  constexpr int64_t HI = static_cast<int64_t>(std::numeric_limits<Target>::max());

  std::array<int64_t, StencilType::size> high_prec_int_coeffs{};
  // range_msb = min(countl_zero(|coeff_i|)) -- equals 64 when all coefficients are zero.
  int32_t range_msb = 64;

  for (size_t i = 0; i < StencilType::size; ++i) {
    HighPrec raw = static_cast<HighPrec>(s.coeffs[i]) / a;
    high_prec_int_coeffs[i] = static_cast<int64_t>(raw + (raw >= HighPrec(0) ? HighPrec(0.5) : HighPrec(-0.5)));
    int64_t v = high_prec_int_coeffs[i];
    // Safe conversion to uint64_t: avoids UB from std::abs(INT64_MIN) and the
    // fact that std::countl_zero only accepts unsigned types in C++20.
    uint64_t abs_val = (v < 0) ? static_cast<uint64_t>(-v) : static_cast<uint64_t>(v);
    if (int32_t msb = std::countl_zero(abs_val); msb < range_msb)
      range_msb = msb;
  }

  // Position (0-indexed from LSB) of the most significant bit across all
  // absolute coefficients.  -1 when every coefficient rounds to zero.
  int32_t p_max = 63 - range_msb;

  StencilSlices<IntStencilType, N> result;
  std::array<int64_t, StencilType::size> residual = high_prec_int_coeffs;

  for (int32_t k = 0; k < N; ++k) {
    // Arithmetic shift that aligns the k-th window of target_bits bits
    // to positions [target_bits-1 : 0].  Consecutive slices are target_bits
    // apart, matching the carry step (>> target_bits) in the GPU kernel.
    // For k=0 the shift is always non-negative when M>=2 (p_max >= 7);
    // max(0,...) protects the M=1 case where p_max < target_bits.
    // For k>0 the shift can be negative, meaning the residual has fewer
    // bits than target_bits and must be left-shifted into the window.
    int32_t shift = p_max - (target_bits - 1) + extra_msb_shift - target_bits * k;
    if (k == 0)
      shift = std::max(0, shift);

    result.slices[k].offsets = s.offsets;

    for (size_t i = 0; i < StencilType::size; ++i) {
      int64_t val = (shift >= 0) ? (residual[i] >> shift) : (residual[i] << (-shift));
      // Clamp to the symmetric int8 range [-127, 127] to keep multiplication
      // well-behaved and to avoid the prohibited INT8_MIN value.
      val = std::max(LO, std::min(HI, val));
      result.slices[k].coeffs[i] = static_cast<Target>(val);
      // Subtract this slice's exact contribution so the residual carries only
      // the bits not yet captured.
      residual[i] -= (shift >= 0)
          ? (static_cast<int64_t>(result.slices[k].coeffs[i]) << shift)
          : (static_cast<int64_t>(result.slices[k].coeffs[i]) >> (-shift));
    }
  }

  // Compute boundary gain: encode round(128^(N-1) / effective_a) in base-128.
  //
  // The GPU kernel's multi-digit multiplication of data (N digits) by
  // boundary_gain (N digits) produces a 2N-1 digit product, but only the
  // upper N digits are stored -- the lower N-1 digits are truncated. This
  // implicit division by 128^(N-1) means the effective per-iteration
  // boundary scaling is: boundary_gain_combined / 128^(N-1).
  //
  // For boundaries and interior to share the same rescaling factor
  // (scaling_factor = 1/effective_a), we need:
  //   boundary_gain_combined / 128^(N-1) = 1/effective_a
  //   boundary_gain_combined = 128^(N-1) / effective_a
  {
    constexpr unsigned __int128 base = static_cast<unsigned __int128>(1) << target_bits; // 128

    // Compute shift0 (same formula as the k=0 slice shift above)
    int32_t shift0 = std::max(0, p_max - (target_bits - 1) + extra_msb_shift);

    // bg_value = 128^(N-1) / effective_a = 128^(N-1) / (a * 2^shift0)
    // Compute in high precision, then round to nearest integer.
    HighPrec eff_a = OzakiTraits<HighPrec>::scalbn(a, shift0);
    HighPrec bg_real = HighPrec(1) / eff_a;
    // Multiply by 128^(N-1) to account for the kernel's implicit LSB truncation
    for (int32_t k = 0; k < N - 1; ++k)
      bg_real *= static_cast<HighPrec>(base);

    // Round to nearest integer using HighPrec arithmetic
    unsigned __int128 bg_value;
    if (bg_real >= HighPrec(0)) {
      bg_value = static_cast<unsigned __int128>(bg_real + HighPrec(0.5));
    } else {
      bg_value = 0; // defensive: should always be positive
    }

    // Decompose into N base-128 digits (LSB first extraction, MSB-first storage)
    for (int32_t si = N - 1; si >= 0; --si) {
      result.boundary_gain[si] = static_cast<Target>(bg_value % base);
      bg_value /= base;
    }

    // Compute high-precision emulated boundary gain with n_emulated digits.
    // Encodes 1/effective_a in as many base-128 digits as there are data slices,
    // giving precision ~1/128^(n_emulated-1). This eliminates per-iteration
    // quantization error at boundary cells.
    result.n_emulated_gain_slices = n_emulated;
    if (n_emulated > 0) {
      HighPrec value = HighPrec(1) / eff_a;
      for (int32_t g = 0; g < n_emulated; ++g) {
        int64_t digit_int = static_cast<int64_t>(
            value + (value >= HighPrec(0) ? HighPrec(0.5) : HighPrec(-0.5)));
        digit_int = std::max(static_cast<int64_t>(LO), std::min(static_cast<int64_t>(HI), digit_int));
        result.emulated_gain[g] = static_cast<Target>(digit_int);
        value = (value - static_cast<HighPrec>(digit_int)) * static_cast<HighPrec>(base);
      }
    }
  }

  result.compute_packed_coeffs();
  return result;
}


template <OzakiPair O> void mend(std::span<typename O::data_type> out, const OzakiSlices<O> &slices) {
  using Data = typename O::data_type;
  using Target = typename O::target_type;

  size_t n = slices.width;
  size_t n_slices = slices.exponents.size();

  if (out.size() != n) {
    throw std::runtime_error("Output buffer is not adequately sized.");
  }

  std::fill(out.begin(), out.end(), static_cast<Data>(0));

  for (size_t i = 0; i < n_slices; ++i) {
    std::span<const Target> sptr = slices.get_slice(i);
    const int32_t scale = slices.exponents[i];

    for (size_t j = 0; j < n; ++j) {
      out[j] += OzakiPairTraits<O>::up_cast(sptr[j], scale);
    }
  }
}

template <std::floating_point T, size_t... Dims>
void stencil(std::span<T> out, const ozi::Stencil<T, Dims...> &s, std::span<const T> in,
             const std::array<size_t, sizeof...(Dims)> &shape) {

  using Traits = OzakiFP_Int8<T>;
  using Data = typename Traits::data_type;
  using Target = typename Traits::target_type;
  using Acc = typename Traits::acc_type;

  size_t domain_size = 1;
  for (auto d : shape)
    domain_size *= d;

  if (out.size() != domain_size || in.size() != domain_size) {
    throw std::runtime_error("Input/Output buffers do not match domain shape size");
  }

  auto input_slices = slice<Traits>(in);

  auto stencil_coeffs_span = std::span<const T>(s.coeffs.data(), s.size);
  auto stencil_slices = slice<Traits>(stencil_coeffs_span);

  std::vector<Acc> temp_acc(domain_size);

  std::fill(out.begin(), out.end(), static_cast<Data>(0));

  using IntStencilType = ozi::Stencil<Target, Dims...>;

  size_t max_slices = std::max(input_slices.exponents.size(), stencil_slices.exponents.size());

  for (size_t sI = 0; sI < input_slices.exponents.size(); ++sI) {
    for (size_t sS = 0; sS < stencil_slices.exponents.size(); ++sS) {

      IntStencilType k_int;
      auto coeff_slice = stencil_slices.get_slice(sS);
      std::copy(coeff_slice.begin(), coeff_slice.end(), k_int.coeffs.begin());

      k_int.offsets = s.offsets;
      k_int.template apply_to<Acc, std::vector<Acc>, std::span<const Target>>(temp_acc, input_slices.get_slice(sI),
                                                                              shape);
      int32_t scale = input_slices.exponents[sI] + stencil_slices.exponents[sS];

      for (size_t i = 0; i < domain_size; ++i) {
        out[i] += OzakiPairTraits<Traits>::up_cast_acc(temp_acc[i], scale, 2);
      }
    }
  }
}

} // namespace ozicpu

#endif // HEADER_OZAKI_HPP
