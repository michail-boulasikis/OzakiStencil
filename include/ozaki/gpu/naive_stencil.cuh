// Direct (non-sliced) stencil kernels in FP32/FP64/FP128: the performance
// baselines of bench-stencil.
#ifndef OZAKI_GPU_NAIVE_STENCIL_CUH
#define OZAKI_GPU_NAIVE_STENCIL_CUH

#include <ozaki/core/stencil.hpp>

#include <cstdint>

namespace ozaki {

// The stencil (a Stencil<T, Dims...> object) is copied here before launch;
// 8 KiB holds the largest one (5x5x5 in FP128).
__constant__ int8_t stencil_cmem[8192];

// One thread per cell (x = threadIdx.x, then y, z); interior cells only.
template <typename T, size_t... Dims>
  requires(sizeof...(Dims) == 2 || sizeof...(Dims) == 3)
__global__ void naive_stencil(T *__restrict__ out, const T *__restrict__ in,
                              const typename Stencil<T, Dims...>::shape_type shape) {
  using S = Stencil<T, Dims...>;
  const S *stencil = reinterpret_cast<const S *>(stencil_cmem);
  const int32_t x = static_cast<int32_t>(threadIdx.x + blockDim.x * blockIdx.x);
  const int32_t y = static_cast<int32_t>(threadIdx.y + blockDim.y * blockIdx.y);
  const int32_t z = static_cast<int32_t>(threadIdx.z + blockDim.z * blockIdx.z);
  constexpr typename S::shape_type extent = {Dims...};
  auto interior = [&](int32_t index, size_t dim) {
    return index >= stencil->offsets[dim] && index <= shape[dim] - (extent[dim] - stencil->offsets[dim]);
  };
  const auto strides = make_strides_from_shape(shape);
  if constexpr (S::dims == 2) {
    if (interior(y, 0) && interior(x, 1)) {
      out[x + y * shape[1]] = S::template apply<T>(stencil->coeffs.data(), stencil->offsets, in, strides, y, x);
    }
  } else {
    if (interior(z, 0) && interior(y, 1) && interior(x, 2)) {
      out[x + y * shape[2] + z * shape[1] * shape[2]] =
          S::template apply<T>(stencil->coeffs.data(), stencil->offsets, in, strides, z, y, x);
    }
  }
}

} // namespace ozaki

#endif // OZAKI_GPU_NAIVE_STENCIL_CUH
