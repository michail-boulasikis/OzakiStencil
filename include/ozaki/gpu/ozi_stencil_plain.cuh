#ifndef HEADER_OZI_STENCIL_PLAIN
#define HEADER_OZI_STENCIL_PLAIN

#include "../core/oztraits.hpp"
#include "../core/stencil.hpp"
#include "ozi_kernels.cuh"

#include <climits>
#include <cuda_runtime.h>
#include <type_traits>

namespace ozigpu {
/** Computes S @ u, where u is the ozaki decomposition of a tensor, and S is a stencil */
template <OzakiPair O, size_t... Dims>
  requires(sizeof...(Dims) <= 3) && std::is_integral_v<typename O::target_type>
__global__ void stencil_v1(typename O::target_type *__restrict__ out, const typename O::target_type *__restrict__ in,
                           const Stencil<typename O::target_type, Dims...> s,
                           const typename Stencil<typename O::target_type, Dims...>::shape_type input_shape,
                           OzakiDescriptor<O> *__restrict__ desc) {
  using Data = typename O::data_type;
  using Target = typename O::target_type;
  using Acc = typename O::acc_type;

  const int32_t total_slices = desc->n_slices_input;
  const int32_t carry_idx = desc->carry_slice_index;
  const Target threshold = desc->overflow_danger_threshold;
  __shared__ int32_t smem[32];

  int32_t i, j, k;
  i = threadIdx.x + blockDim.x * blockIdx.x;
  if constexpr (sizeof...(Dims) > 1)
    j = threadIdx.y + blockDim.y * blockIdx.y;
  if constexpr (sizeof...(Dims) > 2)
    k = threadIdx.z + blockDim.z * blockIdx.z;
  int32_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

  if (tid < 32) {
    smem[tid] = 0;
  }

  bool in_bounds = true;
  const auto &s_shape = s.get_shape();
  auto check_dim = [&] __device__(int32_t idx, size_t dim_idx) {
    return (idx >= s.offsets[dim_idx] && idx <= input_shape[dim_idx] - (s_shape[dim_idx] - s.offsets[dim_idx]));
  };

  if constexpr (sizeof...(Dims) == 1) {
    in_bounds = in_bounds && check_dim(i, 0);
  } else if constexpr (sizeof...(Dims) == 2) {
    in_bounds = in_bounds && check_dim(j, 0);
    in_bounds = in_bounds && check_dim(i, 1);
  } else if constexpr (sizeof...(Dims) == 3) {
    in_bounds = in_bounds && check_dim(k, 0);
    in_bounds = in_bounds && check_dim(j, 1);
    in_bounds = in_bounds && check_dim(i, 2);
  }

  bool valid_thread = (i < input_shape[0]);
  if constexpr (sizeof...(Dims) > 1)
    valid_thread &= (j < input_shape[1]);
  if constexpr (sizeof...(Dims) > 2)
    valid_thread &= (k < input_shape[2]);

  size_t slice_stride = 1;
  for (auto d : input_shape) {
    slice_stride *= d;
  }

  Acc stencil_gain = 0;
  for (const auto &c : s.coeffs) {
    stencil_gain += abs(static_cast<Acc>(c));
  }
  const auto input_strides = ozi::make_strides_from_shape(input_shape);

  Acc local_val = static_cast<Acc>(0);

  if (valid_thread) {
    size_t out_offset = 0;
    if constexpr (sizeof...(Dims) == 1)
      out_offset = i;
    else if constexpr (sizeof...(Dims) == 2)
      out_offset = i + j * input_shape[0];
    else if constexpr (sizeof...(Dims) == 3)
      out_offset = i + j * input_shape[1] + k * input_shape[0] * input_shape[1];

#pragma unroll 4
    for (int32_t logic_s = total_slices - 1; logic_s >= 0; --logic_s) {
      int32_t phys_idx = (carry_idx + logic_s) % total_slices;
      const Target *current_slice_input = in + slice_stride * phys_idx;
      Target *current_slice_output = out + slice_stride * phys_idx;

      Acc partial_result = static_cast<Acc>(0);
      bool treat_as_zero = (logic_s == 0 && desc->just_shifted);

      if (!treat_as_zero) {
        if (in_bounds) {
          if constexpr (sizeof...(Dims) == 1)
            partial_result = s.template apply_at_as<Acc>(current_slice_input, input_strides, i);
          else if constexpr (sizeof...(Dims) == 2)
            partial_result = s.template apply_at_as<Acc>(current_slice_input, input_strides, j, i);
          else if constexpr (sizeof...(Dims) == 3)
            partial_result = s.template apply_at_as<Acc>(current_slice_input, input_strides, k, j, i);
        } else {
          partial_result = static_cast<Acc>(current_slice_input[out_offset]) * stencil_gain;
        }
      }

      local_val += partial_result;

      if (logic_s != 0) {
        Target slice_val = static_cast<Target>(local_val);
        current_slice_output[out_offset] = slice_val;
        local_val = (local_val - static_cast<Acc>(slice_val)) >> (CHAR_BIT * sizeof(Target) - 1);
      } else {
        current_slice_output[out_offset] = static_cast<Target>(local_val);
      }
    }
  }

  __syncthreads();
  int32_t dangerous = __any_sync(__activemask(), (abs(local_val) > static_cast<Acc>(threshold)));
  if (tid % WARP_SIZE == 0)
    smem[tid / WARP_SIZE] = dangerous;
  __syncthreads();

  if (tid < 32) {
    int32_t flag = smem[tid];
    flag = __any_sync(__activemask(), flag);
    if (flag && (__ffs(__activemask()) - 1 == tid)) {
      atomicExch(&desc->shift_required, 1);
    }
  }
}
} // namespace ozigpu

#endif // HEADER_OZI_STENCIL_PLAIN
