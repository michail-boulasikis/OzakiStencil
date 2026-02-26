#ifndef HEADER_OZI_STENCIL_PLAIN
#define HEADER_OZI_STENCIL_PLAIN

#include "../core/oztraits.hpp"
#include "../core/stencil.hpp"
#include "ozi_kernels.cuh"

#include <climits>
#include <cuda_runtime.h>
#include <type_traits>

namespace ozigpu {
/**
 * Computes S @ u, where u is the ozaki decomposition of a tensor, and S is a
 * multi-slice stencil described by M Target-type stencil slices.
 *
 * Template parameter M is the number of stencil slices (compile-time constant).
 * The stencil array s[0..M-1] must be ordered from most significant to least
 * significant slice. Stencil slices do not undergo buffer rotation.
 *
 * The outer loop iterates over data slices (LSB to MSB) to minimize data loads.
 * The inner loop iterates over stencil slices, accumulating pair-wise products
 * into M accumulators of equal scale, with carry propagation between result
 * positions.
 */
template <OzakiPair O, int32_t M, size_t... Dims>
  requires(sizeof...(Dims) <= 3) && std::is_integral_v<typename O::target_type> && (M >= 1)
__global__ void stencil_v1(typename O::target_type *__restrict__ out, const typename O::target_type *__restrict__ in,
                           const ozi::StencilSlices<Stencil<typename O::target_type, Dims...>, M> s,
                           const typename Stencil<typename O::target_type, Dims...>::shape_type input_shape,
                           OzakiDescriptor<O> *__restrict__ desc) {
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

  // Bounds check uses s[0] since all slices share the same shape
  const auto &s_shape = s.slices[0].get_shape();
  auto check_dim = [&] __device__(int32_t idx, size_t dim_idx) {
    return (idx >= s.slices[0].offsets[dim_idx] &&
            idx <= input_shape[dim_idx] - (s_shape[dim_idx] - s.slices[0].offsets[dim_idx]));
  };

  bool in_bounds;
  if constexpr (sizeof...(Dims) == 1) {
    in_bounds = check_dim(i, 0);
  } else if constexpr (sizeof...(Dims) == 2) {
    in_bounds = check_dim(j, 0) && check_dim(i, 1);
  } else {
    in_bounds = check_dim(k, 0) && check_dim(j, 1) && check_dim(i, 2);
  }

  bool valid_thread;
  if constexpr (sizeof...(Dims) == 1)
    valid_thread = (i < input_shape[0]);
  else if constexpr (sizeof...(Dims) == 2)
    valid_thread = (j < input_shape[0]) && (i < input_shape[1]);
  else
    valid_thread = (k < input_shape[0]) && (j < input_shape[1]) && (i < input_shape[2]);

  size_t slice_stride = 1;
  for (auto d : input_shape) {
    slice_stride *= d;
  }

  const auto input_strides = ozi::make_strides_from_shape(input_shape);

  using StencilSlices_type = ozi::StencilSlices<Stencil<Target, Dims...>, M>;

  // M accumulators for the sliding window of result positions.
  // acc[0] tracks the most significant active position,
  // acc[M-1] tracks the least significant (and the next to be output).
  Acc acc[M];
  for (int32_t si = 0; si < M; ++si)
    acc[si] = static_cast<Acc>(0);
  Acc carry = static_cast<Acc>(0);

  if (valid_thread) {
    size_t out_offset = 0;
    if constexpr (sizeof...(Dims) == 1)
      out_offset = i;
    else if constexpr (sizeof...(Dims) == 2)
      out_offset = i + j * input_shape[1];
    else if constexpr (sizeof...(Dims) == 3)
      out_offset = i + j * input_shape[2] + k * input_shape[1] * input_shape[2];

    if (in_bounds) {
      // Main loop: process data slices from LSB to MSB
      // Each iteration loads one data slice and applies all M stencil slices to
      // it. Products S_si @ D_j contribute to result position (j + si), which
      // maps to accumulator index si. After accumulation, the completed result
      // in acc[M-1] is extracted, and the carry propagates via a shift of the
      // accumulator window.
#pragma unroll 4
      for (int32_t logic_s = total_slices - 1; logic_s >= 0; --logic_s) {
        // Shift accumulators: slide the window one position towards MSB.
        // acc[M-1] receives the old acc[M-2] plus carry from previous output.
        for (int32_t si = M - 1; si > 0; --si) {
          acc[si] = acc[si - 1];
        }
        acc[0] = static_cast<Acc>(0);
        acc[M - 1] += carry;
        carry = static_cast<Acc>(0);

        // Load data slice once (the critical path bottleneck)
        int32_t phys_idx = (carry_idx + logic_s) % total_slices;
        const Target *current_slice_input = in + slice_stride * phys_idx;
        bool treat_as_zero = (logic_s == 0 && desc->just_shifted);

        // Inner loop: apply all M stencil slices to the loaded data slice.
        // Iterates from least significant to most significant stencil slice.
        // S_si @ D_{logic_s} contributes to result position (logic_s + si).
        for (int32_t si = M - 1; si >= 0; --si) {
          // Skip if the result position falls outside the valid range
          if (logic_s + si >= total_slices)
            continue;

          Acc partial = static_cast<Acc>(0);
          if (!treat_as_zero) {
            if constexpr (sizeof...(Dims) == 1)
              partial = s.slices[si].template apply_at_as<Acc>(current_slice_input, input_strides, i);
            else if constexpr (sizeof...(Dims) == 2)
              partial = s.slices[si].template apply_at_as<Acc>(current_slice_input, input_strides, j, i);
            else if constexpr (sizeof...(Dims) == 3)
              partial = s.slices[si].template apply_at_as<Acc>(current_slice_input, input_strides, k, j, i);
          }
          acc[si] += partial;
        }

        // Output: the result at position (logic_s + M - 1) is now complete
        // (all contributing S_si @ D_j terms have been accumulated).
        int32_t out_pos = logic_s + M - 1;
        if (out_pos < total_slices) {
          int32_t out_phys = (carry_idx + out_pos) % total_slices;
          Target *current_slice_output = out + slice_stride * out_phys;

          if (out_pos != 0) {
            Target slice_val = static_cast<Target>(acc[M - 1]);
            current_slice_output[out_offset] = slice_val;
            carry = (acc[M - 1] - static_cast<Acc>(slice_val)) >> (CHAR_BIT * sizeof(Target) - 1);
          } else {
            // MSB result: store directly, no carry computation
            current_slice_output[out_offset] = static_cast<Target>(acc[M - 1]);
          }
        }
      }

      // ---- Cleanup: flush remaining result positions ----
      const int32_t cleanup_start = M - 2;
      for (int32_t p = cleanup_start; p >= 0; --p) {
        for (int32_t si = M - 1; si > 0; --si) {
          acc[si] = acc[si - 1];
        }
        acc[0] = static_cast<Acc>(0);
        acc[M - 1] += carry;
        carry = static_cast<Acc>(0);

        if (p < total_slices) {
          int32_t out_phys = (carry_idx + p) % total_slices;
          Target *current_slice_output = out + slice_stride * out_phys;

          if (p != 0) {
            Target slice_val = static_cast<Target>(acc[M - 1]);
            current_slice_output[out_offset] = slice_val;
            carry = (acc[M - 1] - static_cast<Acc>(slice_val)) >> (CHAR_BIT * sizeof(Target) - 1);
          } else {
            // MSB result: store directly
            current_slice_output[out_offset] = static_cast<Target>(acc[M - 1]);
          }
        }
      }

    } else {
      // ========= BOUNDARY PATH: full N×N_bg multiplication with emulated_gain =========
      const int32_t n_bg = s.n_emulated_gain_slices;
      const bool skip_lsb = desc->just_shifted;

      if (n_bg > 0) {
        int32_t boundary_result[StencilSlices_type::MAX_DATA_SLICES] = {};

        for (int32_t d = total_slices - 1; d >= 0; --d) {
          if (d == 0 && skip_lsb)
            continue;
          int32_t phys_d = (carry_idx + d) % total_slices;
          Acc data_d = static_cast<Acc>(in[slice_stride * phys_d + out_offset]);

          for (int32_t g = 0; g < n_bg; ++g) {
            int32_t r = d + g;
            if (r < total_slices)
              boundary_result[r] += static_cast<int32_t>(data_d) * static_cast<int32_t>(s.emulated_gain[g]);
          }
        }

        // Carry propagation and write output
        for (int32_t r = total_slices - 1; r > 0; --r) {
          int32_t phys_r = (carry_idx + r) % total_slices;
          Target digit = static_cast<Target>(boundary_result[r]);
          out[slice_stride * phys_r + out_offset] = digit;
          boundary_result[r - 1] +=
              (boundary_result[r] - static_cast<int32_t>(digit)) >> (CHAR_BIT * sizeof(Target) - 1);
        }
        // MSB: no carry target
        out[slice_stride * ((carry_idx) % total_slices) + out_offset] = static_cast<Target>(boundary_result[0]);

        // Set acc[M-1] for overflow detection
        acc[M - 1] = static_cast<Acc>(boundary_result[0]);
      }
    }
  }

  // Overflow detection: check the MSB accumulator value (before truncation to
  // Target) against the threshold. If any thread in the block has a dangerous
  // value, signal that a shift of the Ozaki representation is required.
  __syncthreads();
  int32_t dangerous = __any_sync(__activemask(), (abs(acc[M - 1]) > static_cast<Acc>(threshold)));
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
