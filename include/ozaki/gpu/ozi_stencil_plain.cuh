#ifndef HEADER_OZI_STENCIL_PLAIN
#define HEADER_OZI_STENCIL_PLAIN

#include "../core/oztraits.hpp"
#include "../core/stencil.hpp"
#include "ozi_kernels.cuh"

#include <climits>
#include <cuda_runtime.h>
#include <type_traits>

namespace ozigpu {

// Constant memory for stencil data. StencilSlices structs are copied here
// before kernel launch so coefficients are loaded on demand via LDC/LDCU
// instead of being copied to the per-thread stack.
__constant__ int8_t stencil_v1_cmem[8192];

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
 *
 * Stencil data is read from __constant__ memory (stencil_v1_cmem). Callers
 * must copy the StencilSlices struct via cudaMemcpyToSymbol before launch.
 */
template <OzakiPair O, int32_t M, size_t... Dims>
  requires(sizeof...(Dims) <= 3) && std::is_integral_v<typename O::target_type> && (M >= 1)
__global__ void stencil_v1(typename O::target_type *__restrict__ out, const typename O::target_type *__restrict__ in,
                           const typename Stencil<typename O::target_type, Dims...>::shape_type input_shape,
                           OzakiDescriptor<O> *__restrict__ desc) {
  using Target = typename O::target_type;
  using Acc = typename O::acc_type;
  using StencilType = Stencil<Target, Dims...>;
  using StencilSlices_type = ozi::StencilSlices<StencilType, M>;

  // Access stencil data directly from constant memory — no struct copy
  const auto *s = reinterpret_cast<const StencilSlices_type *>(stencil_v1_cmem);

  const int32_t total_slices = desc->n_slices_input;
  const int32_t carry_idx = desc->carry_slice_index;
  const Target threshold = desc->overflow_danger_threshold;
  // Row of the pre-computed physical-index LUT for the current carry_idx
  const int32_t *phys_lut = &phys_lut_cmem[carry_idx * PHYS_LUT_MAX_SLICES];
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
  constexpr auto s_shape = StencilType::get_shape();
  auto check_dim = [&] __device__(int32_t idx, size_t dim_idx) {
    return (idx >= s->slices[0].offsets[dim_idx] &&
            idx <= input_shape[dim_idx] - (s_shape[dim_idx] - s->slices[0].offsets[dim_idx]));
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
        int32_t phys_idx = phys_lut[logic_s];
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
            // Load coefficients for slice si on demand from constant memory
            const Target *coeff_si = s->slices[si].coeffs.data();
            const auto &offsets_si = s->slices[si].offsets;
            if constexpr (sizeof...(Dims) == 1)
              partial = StencilType::template apply_at_as_from<Acc>(coeff_si, offsets_si, current_slice_input,
                                                                    input_strides, i);
            else if constexpr (sizeof...(Dims) == 2)
              partial = StencilType::template apply_at_as_from<Acc>(coeff_si, offsets_si, current_slice_input,
                                                                    input_strides, j, i);
            else if constexpr (sizeof...(Dims) == 3)
              partial = StencilType::template apply_at_as_from<Acc>(coeff_si, offsets_si, current_slice_input,
                                                                    input_strides, k, j, i);
          }
          acc[si] += partial;
        }

        // Output: the result at position (logic_s + M - 1) is now complete
        // (all contributing S_si @ D_j terms have been accumulated).
        int32_t out_pos = logic_s + M - 1;
        if (out_pos < total_slices) {
          int32_t out_phys = phys_lut[out_pos];
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
          int32_t out_phys = phys_lut[p];
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
      // Streaming formulation: process output positions from LSB to MSB,
      // fusing accumulation with carry propagation. Uses O(1) stack instead
      // of O(total_slices) for the boundary_result array.
      const int32_t n_bg = s->n_emulated_gain_slices;
      const bool skip_lsb = desc->just_shifted;

      if (n_bg > 0) {
        Acc b_carry = static_cast<Acc>(0);
        Acc b_raw = static_cast<Acc>(0);

        for (int32_t r = total_slices - 1; r >= 0; --r) {
          b_raw = b_carry;
          b_carry = static_cast<Acc>(0);

          for (int32_t g = 0; g < n_bg; ++g) {
            int32_t d = r - g;
            if (d >= 0 && !(d == 0 && skip_lsb)) {
              int32_t phys_d = phys_lut[d];
              b_raw += static_cast<Acc>(in[slice_stride * phys_d + out_offset])
                     * static_cast<int32_t>(s->emulated_gain[g]);
            }
          }

          int32_t phys_r = phys_lut[r];
          Target *out_ptr = out + slice_stride * phys_r;
          if (r > 0) {
            Target digit = static_cast<Target>(b_raw);
            out_ptr[out_offset] = digit;
            b_carry = (b_raw - static_cast<Acc>(digit)) >> (CHAR_BIT * sizeof(Target) - 1);
          } else {
            out_ptr[out_offset] = static_cast<Target>(b_raw);
          }
        }

        // Set acc[M-1] for overflow detection (MSB value before truncation)
        acc[M - 1] = b_raw;
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
