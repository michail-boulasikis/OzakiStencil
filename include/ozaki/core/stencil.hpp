// A stencil with compile-time shape, used by the CPU reference solver and the
// naive GPU baselines. Coefficients are row-major; `offsets` is the position
// of the center cell.
#ifndef OZAKI_CORE_STENCIL_HPP
#define OZAKI_CORE_STENCIL_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>

#ifdef __CUDACC__
#define OZAKI_HD __host__ __device__
#define OZAKI_HD_INLINE __host__ __device__ __forceinline__
#else
#define OZAKI_HD
#define OZAKI_HD_INLINE inline __attribute__((always_inline))
#endif

namespace ozaki {

template <size_t D> OZAKI_HD constexpr std::array<size_t, D> make_strides_from_shape(const std::array<size_t, D> &shape) {
  std::array<size_t, D> strides{};
  size_t stride = 1;
  for (size_t i = D; i-- > 0;) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

template <typename T, size_t... Dims> struct Stencil {
  static constexpr size_t dims = sizeof...(Dims);
  static constexpr size_t size = (Dims * ...);
  using shape_type = std::array<size_t, dims>;
  static constexpr shape_type shape = {Dims...};
  static constexpr shape_type strides = make_strides_from_shape(shape);

  std::array<T, size> coeffs{};
  shape_type offsets{};

  // sum_i coeffs[i] * src[center + (position of coefficient i - offsets)],
  // with the coefficients read from `coeffs` (e.g. in constant memory).
  template <typename Acc, typename Input, typename... Idx>
  OZAKI_HD_INLINE static Acc apply(const T *__restrict__ coeffs, const shape_type &offsets,
                                   const Input *__restrict__ src, const shape_type &src_strides, Idx... center) {
    const shape_type c = {static_cast<size_t>(center)...};
    int64_t base = 0;
    for (size_t d = 0; d < dims; ++d) {
      base += static_cast<int64_t>((c[d] - offsets[d]) * src_strides[d]);
    }
    return apply_unrolled<Acc>(coeffs, src + base, src_strides, std::make_index_sequence<size>{});
  }

  // output = stencil(input) on the interior of a tensor of shape `in_shape`;
  // boundary cells of `output` are left untouched.
  template <typename Acc, typename Out, typename In>
  void apply_to(Out &output, const In &input, const shape_type &in_shape) const {
    assert(static_cast<const void *>(std::data(output)) != static_cast<const void *>(std::data(input)));
    const shape_type in_strides = make_strides_from_shape(in_shape);
    size_t elements = 1;
    for (const size_t extent : in_shape)
      elements *= extent;
#pragma omp parallel for
    for (size_t linear = 0; linear < elements; ++linear) {
      shape_type idx{};
      bool interior = true;
      for (size_t d = 0; d < dims; ++d) {
        idx[d] = (linear / in_strides[d]) % in_shape[d];
        interior = interior && idx[d] >= offsets[d] && idx[d] + (shape[d] - offsets[d]) <= in_shape[d];
      }
      if (interior) {
        output[linear] = [&]<size_t... D>(std::index_sequence<D...>) {
          return apply<Acc>(coeffs.data(), offsets, std::data(input), in_strides, idx[D]...);
        }(std::make_index_sequence<dims>{});
      }
    }
  }

private:
  template <typename Acc, typename Input, size_t... Is>
  OZAKI_HD_INLINE static Acc apply_unrolled(const T *__restrict__ coeffs, const Input *__restrict__ corner,
                                            const shape_type &src_strides, std::index_sequence<Is...>) {
    return ((static_cast<Acc>(coeffs[Is]) * static_cast<Acc>(corner[offset_of<Is>(src_strides)])) + ...);
  }
  // Offset of coefficient Is from the stencil's corner in the source tensor.
  template <size_t Is> OZAKI_HD_INLINE static int64_t offset_of(const shape_type &src_strides) {
    constexpr shape_type extent = {Dims...}; // local copies are usable in device code
    constexpr shape_type stride = make_strides_from_shape(extent);
    int64_t offset = 0;
    for (size_t d = 0; d < dims; ++d) {
      offset += static_cast<int64_t>((Is / stride[d]) % extent[d]) * static_cast<int64_t>(src_strides[d]);
    }
    return offset;
  }
};

template <typename S, typename NewT> struct RebindStencil;
template <typename T, size_t... Dims, typename NewT> struct RebindStencil<Stencil<T, Dims...>, NewT> {
  using type = Stencil<NewT, Dims...>;
};
template <typename S, typename NewT> using RebindStencil_t = typename RebindStencil<S, NewT>::type;

} // namespace ozaki

#endif // OZAKI_CORE_STENCIL_HPP
