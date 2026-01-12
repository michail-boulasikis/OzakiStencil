#ifndef HEADER_STENCIL_HPP
#define HEADER_STENCIL_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <span>
#include <utility>
#include <tuple>

#ifdef __CUDACC__
#define HD __host__ __device__
#define OZI_ALWAYS_INLINE __forceinline__
#else
#define HD
#if defined(_MSC_VER)
#define OZI_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define OZI_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define OZI_ALWAYS_INLINE inline
#endif
#endif

namespace ozi {

template <size_t D> HD constexpr std::array<size_t, D> make_strides_from_shape(const std::array<size_t, D> &shape) {
  std::array<size_t, D> strides{};
  size_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(D) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = acc;
    acc *= shape[static_cast<size_t>(i)];
  }
  return strides;
}

template <size_t D>
HD constexpr size_t linear_from_multi(const std::array<size_t, D> &idx, const std::array<size_t, D> &strides) {
  size_t lin = 0;
  for (size_t d = 0; d < D; ++d)
    lin += idx[d] * strides[d];
  return lin;
}

template <size_t D>
HD constexpr std::array<size_t, D> multi_from_linear(size_t linear, const std::array<size_t, D> &shape) {
  std::array<size_t, D> idx{};
  auto strides = make_strides_from_shape(shape);
  for (size_t i = 0; i < D; ++i) {
    idx[i] = (linear / strides[i]) % shape[i];
  }
  return idx;
}

template <typename T, size_t... Dims> HD struct Stencil {
  static_assert(sizeof...(Dims) >= 1, "Stencil must have at least one dimension");
  static_assert(((Dims > 0) && ...), "All stencil dimensions must be > 0");

public:
  static constexpr size_t dims = sizeof...(Dims);
  HD static constexpr size_t get_dims() { return dims; }

  using value_type = T;
  using shape_type = std::array<size_t, dims>;
  using offset_type = std::array<size_t, dims>;

  inline static constexpr shape_type shape = {Dims...};
  HD static constexpr shape_type get_shape() { return {Dims...}; }
  inline static constexpr size_t size = (Dims * ...);
  HD static constexpr size_t get_size() { return (Dims * ...); }

  using coeffs_type = std::array<T, size>;

  inline static constexpr shape_type strides = make_strides_from_shape(shape);
  HD static constexpr shape_type get_strides() { return make_strides_from_shape(shape_type{Dims...}); }

  coeffs_type coeffs{};
  offset_type offsets{};

  HD constexpr Stencil() noexcept : coeffs{}, offsets{} {
    for (size_t i = 0; i < dims; ++i)
      offsets[i] = shape[i] / 2;
  }

  HD constexpr Stencil(const coeffs_type &c, const offset_type &offs) noexcept : coeffs(c), offsets(offs) {}

  HD constexpr explicit Stencil(const coeffs_type &c) noexcept : coeffs(c), offsets{} {
    for (size_t i = 0; i < dims; ++i)
      offsets[i] = shape_type{shape}[i] / 2;
  }

  HD constexpr explicit Stencil(std::span<const T> c) : offsets{} {
    std::memcpy(coeffs.data(), &c, sizeof(T) * size);
    for (size_t i = 0; i < dims; ++i)
      offsets[i] = shape_type{shape}[i] / 2;
  }

  HD static constexpr size_t multi_to_linear(const shape_type &idx) {
    size_t lin = 0;
    for (size_t i = 0; i < dims; ++i)
      lin += idx[i] * get_strides()[i];
    return lin;
  }

  HD static constexpr shape_type linear_to_multi(int64_t linear) {
    shape_type idx{};
    for (size_t i = 0; i < dims; ++i) {
      idx[i] = (linear / get_strides()[i]) % get_shape()[i];
    }
    return idx;
  }

  template <typename... Idx>
    requires(sizeof...(Idx) == dims)
  HD constexpr T &operator()(Idx... idxs) {
    shape_type idx = {static_cast<size_t>(idxs)...};
    return coeffs[multi_to_linear(idx)];
  }
  template <typename... Idx>
    requires(sizeof...(Idx) == dims)
  HD constexpr const T &operator()(Idx... idxs) const {
    shape_type idx = {static_cast<size_t>(idxs)...};
    return coeffs[multi_to_linear(idx)];
  }

  // these two functions overflow and yet they work. I need a smarter way to handle them.
  template <typename... Idx>
    requires(sizeof...(Idx) == dims)
  HD constexpr const T &cfrel(Idx... relidxs) const noexcept {
    std::array<size_t, dims> rel = {static_cast<size_t>(relidxs)...};
    shape_type abs{};
    for (size_t i = 0; i < dims; ++i) {
      abs[i] = offsets[i] + rel[i];
    }
    return coeffs[multi_to_linear(abs)];
  }
  template <typename... Idx>
    requires(sizeof...(Idx) == dims)
  HD constexpr T &cfrel(Idx... relidxs) noexcept {
    std::array<size_t, dims> rel = {static_cast<size_t>(relidxs)...};
    shape_type abs{};
    for (size_t i = 0; i < dims; ++i) {
      abs[i] = offsets[i] + rel[i];
    }
    return coeffs[multi_to_linear(abs)];
  }

  HD constexpr T &operator[](size_t linear) noexcept { return coeffs[linear]; }
  HD constexpr const T &operator[](size_t linear) const noexcept { return coeffs[linear]; }

  template <size_t Is> HD constexpr int64_t compute_neighbor_offset(const std::array<size_t, dims> &in_strides) const {
    constexpr auto s_idx = linear_to_multi(Is);
    int64_t offset = 0;

    for (size_t d = 0; d < dims; ++d) {
      int64_t diff = static_cast<int64_t>(s_idx[d]) - static_cast<int64_t>(offsets[d]);
      offset += diff * static_cast<int64_t>(in_strides[d]);
    }
    return offset;
  }
  template <typename AccType, typename InputT, typename... Idx, size_t... Is>
  HD OZI_ALWAYS_INLINE AccType apply_unrolled_impl(const InputT *__restrict__ src,
                                                   const std::array<size_t, dims> &in_strides,
                                                   std::index_sequence<Is...>, Idx... center_indices) const {

    const std::array<size_t, dims> c_idx = {static_cast<size_t>(center_indices)...};
    int64_t center_base = 0;
    for (size_t d = 0; d < dims; ++d) {
      center_base += static_cast<int64_t>(c_idx[d]) * static_cast<int64_t>(in_strides[d]);
    }

    // fold expression with casting to AccType
    return ((static_cast<AccType>(coeffs[Is]) *
             static_cast<AccType>(src[center_base + compute_neighbor_offset<Is>(in_strides)])) +
            ...);
  }
  template <typename InputT, typename... Idx>
    requires(sizeof...(Idx) == dims)
  HD OZI_ALWAYS_INLINE auto apply_at(const InputT *__restrict__ src, const std::array<size_t, dims> &in_strides,
                                     Idx... idx) const {
    using CommonType = std::common_type_t<T, InputT>;
    return apply_unrolled_impl<CommonType>(src, in_strides, std::make_index_sequence<size>{}, idx...);
  }

  template <typename AccType, typename InputT, typename... Idx>
    requires(sizeof...(Idx) == dims)
  HD OZI_ALWAYS_INLINE AccType apply_at_as(const InputT *__restrict__ src, const std::array<size_t, dims> &in_strides,
                                           Idx... idx) const {
    return apply_unrolled_impl<AccType>(src, in_strides, std::make_index_sequence<size>{}, idx...);
  }

  template <typename AccType, typename OutContainer, typename InContainer>
  void apply_to(OutContainer &output, const InContainer &input, const shape_type &in_shape) const {

    if constexpr (requires {
                    std::data(output);
                    std::data(input);
                  }) {
      const void *out_ptr = static_cast<const void *>(std::data(output));
      const void *in_ptr = static_cast<const void *>(std::data(input));
      if (out_ptr && in_ptr) {
        assert(out_ptr != in_ptr && "Cannot apply stencil in-place");
      }
    }

    const auto in_strides = make_strides_from_shape(in_shape);
    size_t in_size_lin = 1;
    for (const auto sz : in_shape)
      in_size_lin *= sz;

    auto linear_to_multi_fast = [&](size_t linear) {
      std::array<size_t, dims> idx{};
      for (size_t i = 0; i < dims; ++i) {
        idx[i] = (linear / in_strides[i]) % in_shape[i];
      }
      return idx;
    };

    for (size_t lin = 0; lin < in_size_lin; ++lin) {
      auto idx = linear_to_multi_fast(lin);
      bool valid = true;
#pragma unroll
      for (size_t j = 0; j < dims; ++j)
        valid &= (idx[j] >= offsets[j]) && (idx[j] <= in_shape[j] - (shape[j] - offsets[j]));
      if (!valid)
        continue;
        output[lin] = std::apply([&](auto... unpacked_indices) {
          return this->apply_at_as<AccType>(std::data(input), in_strides, unpacked_indices...);
        }, idx);
    }
  }
};

template <typename T1, size_t... A, typename T2, size_t... B>
HD constexpr auto correlate(const Stencil<T1, A...> &a, const Stencil<T2, B...> &b)
    -> Stencil<std::common_type_t<T1, T2>, (A + B - 1)...> {
  static_assert(sizeof...(A) == sizeof...(B), "Both stencils must have the same number of dimensions");

  using out_type = std::common_type_t<T1, T2>;
  using out_stencil_type = Stencil<out_type, (A + B - 1)...>;
  using out_coeffs_type = typename out_stencil_type::coeffs_type;
  using out_shape_type = typename out_stencil_type::shape_type;

  out_stencil_type out{};
  for (size_t i = 0; i < out_stencil_type::shape.size(); ++i) {
    out.offsets[i] = a.offsets[i] + b.offsets[i];
  }

  constexpr int64_t sizeA = Stencil<T1, A...>::size;
  constexpr int64_t sizeB = Stencil<T2, B...>::size;

  for (size_t ia = 0; ia < sizeA; ++ia) {
    auto idxA = a.linear_to_multi(ia);
    for (size_t jb = 0; jb < sizeB; ++jb) {
      auto idxB = b.linear_to_multi(jb);
      out_shape_type out_idx{};
      for (size_t d = 0; d < out_idx.size(); ++d) {
        out_idx[d] = idxA[d] + idxB[d];
      }
      const size_t out_lin = out_stencil_type::multi_to_linear(out_idx);
      out.coeffs[out_lin] += static_cast<out_type>(a.coeffs[ia]) * static_cast<out_type>(b.coeffs[jb]);
    }
  }

  return out;
}
} // namespace ozi

#endif // HEADER_STENCIL_HPP
