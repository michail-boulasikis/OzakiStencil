#ifndef HEADER_STENCIL_GPU_CUH
#define HEADER_STENCIL_GPU_CUH

#include "../core/stencil.hpp"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

namespace ozi {

// Constant memory buffer for the stencil data.
// 8KB is sufficient for the largest supported stencil (Stencil<__float128, 7, 7, 7> ~ 5.5KB).
__constant__ int8_t stencil_cmem[8192];

template <size_t... Dims>
  requires(sizeof...(Dims) <= 3)
__global__ void stencil_fp32(float *__restrict__ out, const float *__restrict__ in,
                             const typename Stencil<float, Dims...>::shape_type input_shape) {

  int32_t i, j, k;
  i = threadIdx.x + blockDim.x * blockIdx.x;
  if constexpr (sizeof...(Dims) > 1)
    j = threadIdx.y + blockDim.y * blockIdx.y;
  if constexpr (sizeof...(Dims) > 2)
    k = threadIdx.z + blockDim.z * blockIdx.z;

  auto s = *reinterpret_cast<const Stencil<float, Dims...> *>(stencil_cmem);

  constexpr auto s_shape = s.get_shape();
  auto check_dim = [&] __device__(int32_t idx, int32_t dim_idx) {
    return (idx >= s.offsets[dim_idx] && idx <= input_shape[dim_idx] - (s_shape[dim_idx] - s.offsets[dim_idx]));
  };

  bool in_bounds;
  if constexpr (sizeof...(Dims) == 1) {
    in_bounds = check_dim(i, 0);
  } else if constexpr (sizeof...(Dims) == 2) {
    in_bounds = check_dim(j, 0) && check_dim(i, 1);
  } else {
    in_bounds = check_dim(k, 0) && check_dim(j, 1) && check_dim(i, 2);
  }

  if (!in_bounds)
    return;

  const auto input_strides = make_strides_from_shape(input_shape);

  size_t out_offset = 0;
  if constexpr (sizeof...(Dims) == 1) {
    out_offset = i;
    out[out_offset] = s.template apply_at_as<float>(in, input_strides, i);
  } else if constexpr (sizeof...(Dims) == 2) {
    out_offset = i + j * input_shape[1];
    out[out_offset] = s.template apply_at_as<float>(in, input_strides, j, i);
  } else if constexpr (sizeof...(Dims) == 3) {
    out_offset = i + j * input_shape[2] + k * input_shape[1] * input_shape[2];
    out[out_offset] = s.template apply_at_as<float>(in, input_strides, k, j, i);
  }
}

template <size_t... Dims>
  requires(sizeof...(Dims) <= 3)
__global__ void stencil_fp64(double *__restrict__ out, const double *__restrict__ in,
                             const typename Stencil<double, Dims...>::shape_type input_shape) {

  int32_t i, j, k;
  i = threadIdx.x + blockDim.x * blockIdx.x;
  if constexpr (sizeof...(Dims) > 1)
    j = threadIdx.y + blockDim.y * blockIdx.y;
  if constexpr (sizeof...(Dims) > 2)
    k = threadIdx.z + blockDim.z * blockIdx.z;

  auto s = *reinterpret_cast<const Stencil<double, Dims...> *>(stencil_cmem);

  constexpr auto s_shape = s.get_shape();
  auto check_dim = [&] __device__(int32_t idx, int32_t dim_idx) {
    return (idx >= s.offsets[dim_idx] && idx <= input_shape[dim_idx] - (s_shape[dim_idx] - s.offsets[dim_idx]));
  };

  bool in_bounds;
  if constexpr (sizeof...(Dims) == 1) {
    in_bounds = check_dim(i, 0);
  } else if constexpr (sizeof...(Dims) == 2) {
    in_bounds = check_dim(j, 0) && check_dim(i, 1);
  } else {
    in_bounds = check_dim(k, 0) && check_dim(j, 1) && check_dim(i, 2);
  }

  if (!in_bounds)
    return;

  const auto input_strides = make_strides_from_shape(input_shape);

  size_t out_offset = 0;
  if constexpr (sizeof...(Dims) == 1) {
    out_offset = i;
    out[out_offset] = s.template apply_at_as<double>(in, input_strides, i);
  } else if constexpr (sizeof...(Dims) == 2) {
    out_offset = i + j * input_shape[1];
    out[out_offset] = s.template apply_at_as<double>(in, input_strides, j, i);
  } else if constexpr (sizeof...(Dims) == 3) {
    out_offset = i + j * input_shape[2] + k * input_shape[1] * input_shape[2];
    out[out_offset] = s.template apply_at_as<double>(in, input_strides, k, j, i);
  }
}

#if !defined(__CUDA_ARCH_LIST__) || __CUDA_ARCH_LIST__ >= 1000
template <size_t... Dims>
  requires(sizeof...(Dims) <= 3)
__global__ void stencil_fp128(__float128 *__restrict__ out, const __float128 *__restrict__ in,
                              const typename Stencil<__float128, Dims...>::shape_type input_shape) {

  int32_t i, j, k;
  i = threadIdx.x + blockDim.x * blockIdx.x;
  if constexpr (sizeof...(Dims) > 1)
    j = threadIdx.y + blockDim.y * blockIdx.y;
  if constexpr (sizeof...(Dims) > 2)
    k = threadIdx.z + blockDim.z * blockIdx.z;

  auto s = *reinterpret_cast<const Stencil<__float128, Dims...> *>(stencil_cmem);

  constexpr auto s_shape = s.get_shape();
  auto check_dim = [&] __device__(int32_t idx, int32_t dim_idx) {
    return (idx >= s.offsets[dim_idx] && idx <= input_shape[dim_idx] - (s_shape[dim_idx] - s.offsets[dim_idx]));
  };

  bool in_bounds;
  if constexpr (sizeof...(Dims) == 1) {
    in_bounds = check_dim(i, 0);
  } else if constexpr (sizeof...(Dims) == 2) {
    in_bounds = check_dim(j, 0) && check_dim(i, 1);
  } else {
    in_bounds = check_dim(k, 0) && check_dim(j, 1) && check_dim(i, 2);
  }

  if (!in_bounds)
    return;

  const auto input_strides = make_strides_from_shape(input_shape);

  size_t out_offset = 0;
  if constexpr (sizeof...(Dims) == 1) {
    out_offset = i;
    out[out_offset] = s.template apply_at_as<__float128>(in, input_strides, i);
  } else if constexpr (sizeof...(Dims) == 2) {
    out_offset = i + j * input_shape[1];
    out[out_offset] = s.template apply_at_as<__float128>(in, input_strides, j, i);
  } else if constexpr (sizeof...(Dims) == 3) {
    out_offset = i + j * input_shape[2] + k * input_shape[1] * input_shape[2];
    out[out_offset] = s.template apply_at_as<__float128>(in, input_strides, k, j, i);
  }
}
#endif

} // namespace ozi

#endif // HEADER_STENCIL_GPU_CUH
