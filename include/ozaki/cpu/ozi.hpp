#ifndef HEADER_OZAKI_HPP
#define HEADER_OZAKI_HPP

#include "../core/ozconcepts.hpp"
#include "../core/stencil.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <concepts>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ozicpu {
using namespace ozi;

// generic unimplemented template which is specialized
template <typename T> int32_t ozaki_scaling_factor(int32_t x) {
  static_assert(false, "`ozaki_scaling_factor` is not implemented for this data type.");
  return -1;
}
template <OzakiPair O> struct OzakiSlices {
  // store all slices contiguously
  std::vector<typename O::target_type> values;
  std::vector<int32_t> exponents;
  size_t width;

  // helper to view a specific slice row without copying
  std::span<const typename O::target_type> get_slice(size_t index) const {
    return {values.data() + (index * width), width};
  }
};

// Supported ozaki schemes
template <std::floating_point T>
  requires(sizeof(T) > 1)
struct OzakiFP_Int8 {
  using data_type = T;
  using target_type = int8_t;
  using acc_type = int32_t;
};

void matmul(int32_t *__restrict C, const int8_t *__restrict A, const int8_t *__restrict B, size_t m, size_t n,
            size_t k);

template <typename T>
  requires std::is_integral_v<T>
inline int32_t ozaki_scaling_factor(int32_t v) {
  return (CHAR_BIT * sizeof(T) - 1) - v;
}

template <OzakiData T, OzakiTarget U>
  requires std::is_floating_point_v<T> && std::is_integral_v<U>
inline U down_cast(T a, int32_t scale) {
  using std::llround;
  using std::scalbn;
  constexpr auto LO = static_cast<int64_t>(std::numeric_limits<U>::min() + 1);
  constexpr auto HI = static_cast<int64_t>(std::numeric_limits<U>::max());
  auto r = llround(scalbn(a, ozaki_scaling_factor<U>(scale)));
  if (r < LO)
    r = LO;
  if (r > HI)
    r = HI;
  return static_cast<U>(r);
}

template <OzakiData T, OzakiTarget U>
  requires std::is_floating_point_v<T> && std::is_integral_v<U>
inline T up_cast(U a, int32_t scale) {
  using std::scalbn;
  return scalbn(static_cast<T>(a), -ozaki_scaling_factor<U>(scale));
}

template <std::floating_point T, std::integral U, std::integral A>
inline T up_cast_acc(A a, int32_t scale, int32_t n_mul) {
  using std::scalbn;
  return scalbn(static_cast<T>(a), -(n_mul * (CHAR_BIT * sizeof(U) - 1) - scale));
}

// generic ozaki scheme routines

template <OzakiPair O> OzakiSlices<O> slice(std::span<const typename O::data_type> data, size_t max_slices = 0) {
  using Data = typename O::data_type;
  using Target = typename O::target_type;
  using std::abs;
  using std::frexp;
  using std::isfinite;
  using std::max;

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
      abs_max = max(abs(val), abs_max);

    if (!isfinite(abs_max))
      throw std::runtime_error("Ozaki scheme diverged: infinity detected.");
    if (abs_max == static_cast<Data>(0))
      break;

    int32_t exp;
    frexp(abs_max, &exp);
    result.exponents.push_back(exp);

    size_t start_offset = result.values.size();
    result.values.resize(start_offset + n);

    Target *current_slice_ptr = result.values.data() + start_offset;
    Data *residual_ptr = residual.data();

    for (size_t j = 0; j < n; ++j) {
      current_slice_ptr[j] = down_cast<Data, Target>(residual_ptr[j], exp);
      residual_ptr[j] -= up_cast<Data, Target>(current_slice_ptr[j], exp);
    }

    current_slice_idx++;
  }

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
      out[j] += up_cast<Data, Target>(sptr[j], scale);
    }
  }
}

template <OzakiPair O>
void mul(std::span<typename O::data_type> out, const OzakiSlices<O> &a, const OzakiSlices<O> &b, size_t m, size_t n,
         size_t k) {
  using std::max;
  using Data = typename O::data_type;
  using Target = typename O::target_type;
  using Accumulator = typename O::acc_type;

  if (out.size() != m * n) {
    throw std::runtime_error("Output buffer is not adequately sized.");
  }

  std::vector<Accumulator> temp(m * n);
  std::fill(out.begin(), out.end(), static_cast<Data>(0));
  size_t max_slices = max(a.exponents.size(), b.exponents.size());

  for (size_t sA = 0; sA < a.exponents.size(); ++sA)
    for (size_t sB = 0; sB < b.exponents.size(); ++sB) {
      if (sA + sB + 1 <= max_slices) {
        matmul(temp.data(), a.get_slice(sA).data(), b.get_slice(sB).data(), m, n, k);
        for (size_t i = 0; i < m; ++i)
          for (size_t j = 0; j < n; ++j)
            out[i + j * m] +=
                up_cast_acc<Data, Target, Accumulator>(temp[i + j * m], a.exponents[sA] + b.exponents[sB], 2);
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
        out[i] += up_cast_acc<Data, Target, Acc>(temp_acc[i], scale, 2);
      }
    }
  }
}

} // namespace ozicpu

#endif // HEADER_OZAKI_HPP
