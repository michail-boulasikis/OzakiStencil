#ifndef HEADER_OZI_STENCIL_CONV
#define HEADER_OZI_STENCIL_CONV

#include "../core/oztraits.hpp"
#include "ozi.cuh"
#include "ozi_kernels.cuh"

#include <climits>
#include <concepts>
#include <cstring>
#include <cuda_runtime.h>
#include <mma.h>
#include <type_traits>

namespace ozigpu {
namespace convstencil {

// ========== Architecture constants (hardcoded for 7×7 stencil) ==========
constexpr int32_t MMA_M = 32;
constexpr int32_t MMA_N = 8;
constexpr int32_t MMA_K = 16;

constexpr int32_t HALO = 3;
constexpr int32_t STENCIL_SIDE = 7;
constexpr int32_t STRIDE = STENCIL_SIDE + 1; // = 8 (power of 2 for 7×7)

constexpr int32_t NUM_WARPS = 4;
constexpr int32_t MMA_NUM = (STENCIL_SIDE * STENCIL_SIDE + MMA_K - 1) / MMA_K; // = 4

// Tile dimensions: output tile = TILE_SIZE_ROW x TILE_SIZE_COL
// Data tile = output tile + 2 x HALO in each direction
constexpr int32_t TILE_SIZE_ROW = NUM_WARPS * MMA_N;             // = 32
constexpr int32_t TILE_SIZE_COL = MMA_M * MMA_N;                 // = 256
constexpr int32_t DATA_TILE_SIZE_ROW = TILE_SIZE_ROW + 2 * HALO; // = 38
constexpr int32_t DATA_TILE_SIZE_COL = TILE_SIZE_COL + 2 * HALO; // = 262

// Stencil2row shared memory dimensions (col-major: sm_row is fast dim)
// SM_SIZE_ROW = number of column groups = floor(DATA_TILE_SIZE_COL / STRIDE) = MMA_M = 32
// SM_SIZE_COL = K-extent covering all output rows' stencil windows + MMA tile padding
constexpr int32_t SM_SIZE_ROW = DATA_TILE_SIZE_COL / STRIDE; // = 32 (logical width)
constexpr int32_t SM_LD = 48; // physical leading dim (padding experiments, must be multiple of 16 for WMMA)
constexpr int32_t SM_SIZE_COL = (TILE_SIZE_ROW - 1) * STENCIL_SIDE + MMA_NUM * MMA_K; // = 281

constexpr int32_t COEFF_LD = MMA_N;                             // = 8 (weight matrix leading dim)
constexpr int32_t WT_SIZE = MMA_NUM * MMA_K * COEFF_LD;         // = 512 (per-half weight size)
constexpr int32_t ACC_TILE_SIZE = MMA_M * MMA_N;                // = 256 (elements per MMA tile)
constexpr int32_t ELEMS_PER_THREAD = ACC_TILE_SIZE / WARP_SIZE; // = 8
constexpr int32_t TRASH = SM_SIZE_COL * SM_LD - 1;              // dirty-bits sentinel index
constexpr int32_t MAX_M = 12;                                   // max supported stencil slices
constexpr int32_t MAX_DATA_SLICES = 24;                         // matches StencilSlices::MAX_DATA_SLICES

// Chunking constants for 2-row processing (reduces register pressure)
constexpr int32_t ROWS_PER_CHUNK = 2;
constexpr int32_t NUM_CHUNKS = MMA_N / ROWS_PER_CHUNK; // = 4
static_assert(MMA_N % ROWS_PER_CHUNK == 0, "MMA_N must be divisible by ROWS_PER_CHUNK");

// Constant memory for weight matrices: [M][2][WT_SIZE] layout (unused currently)
__constant__ int8_t cs_param_matrix_d[MAX_M * 2 * WT_SIZE];

// ========== Boundary gain data (passed by value to kernel) ==========
struct BoundaryGainData {
  int8_t emulated_gain[MAX_DATA_SLICES];
  int32_t n_slices;
};

// ========== On-the-fly LUT computation ==========
// These replace global memory LUT reads with register-only ALU ops.
// Valid only for STRIDE = 8 (power of 2); for other stencil sizes, integer
// division/modulus would be needed.

/** Compute the sm_data[0] (tessellation A / upper) index for data element at
 *  (data_row, data_col) within the data tile. Returns TRASH for invalid positions. */
__device__ __forceinline__ int32_t compute_lut_a(int32_t data_row, int32_t data_col) {
  bool valid = (((data_col + 1) & (STRIDE - 1)) != 0) && (data_col < DATA_TILE_SIZE_COL - 2 * HALO - 1);
  int32_t sm_row = data_col >> 3;                                       // data_col / STRIDE
  int32_t sm_col = STENCIL_SIDE * data_row + (data_col & (STRIDE - 1)); // data_col % STRIDE
  return valid ? (sm_col * SM_LD + sm_row) : TRASH;
}

/** Compute the sm_data[1] (tessellation B / lower) index for data element at
 *  (data_row, data_col) within the data tile. Returns TRASH for invalid positions. */
__device__ __forceinline__ int32_t compute_lut_b(int32_t data_row, int32_t data_col) {
  bool valid = (((data_col + 2) & (STRIDE - 1)) != 0) && (data_col > 2 * HALO);
  int32_t j_shifted = data_col - STENCIL_SIDE;
  int32_t sm_row = j_shifted >> 3;                                       // j_shifted / STRIDE
  int32_t sm_col = STENCIL_SIDE * data_row + (j_shifted & (STRIDE - 1)); // j_shifted % STRIDE
  return valid ? (sm_col * SM_LD + sm_row) : TRASH;
}

// ========== Weight matrix builder for M stencil slices ==========

/** Build the dual-tessellation weight matrices for M stencil slices.
 *
 *  @param param_out  Output buffer: [M * 2 * WT_SIZE] bytes, zero-initialized internally.
 *  @param coeffs     Flat input: [M * STENCIL_SIDE²] int8_t, slice 0 (MSB) first,
 *                    each slice in row-major stencil order.
 */
template <int32_t M> inline void build_param_matrix(int8_t *param_out, const int8_t *coeffs) {
  constexpr int32_t N_COEFFS = STENCIL_SIDE * STENCIL_SIDE;
  std::memset(param_out, 0, M * 2 * WT_SIZE);

  for (int32_t si = 0; si < M; ++si) {
    const int8_t *sc = coeffs + si * N_COEFFS;
    for (int half = 0; half < 2; ++half) {
      for (int i = 0; i < STENCIL_SIDE; ++i) {
        for (int j = 0; j < STENCIL_SIDE; ++j) {
          int k_idx = i * STENCIL_SIDE + j;
          for (int col = 0; col < COEFF_LD; ++col) {
            int src_col;
            bool valid;
            if (half == 0) {
              // Upper triangle: j >= col -> src_col = j - col
              valid = (j >= col);
              src_col = j - col;
            } else {
              // Lower triangle: j < col -> src_col = j - col + STENCIL_SIDE
              valid = (j < col);
              src_col = j - col + STENCIL_SIDE;
            }
            if (valid && src_col >= 0 && src_col < STENCIL_SIDE) {
              param_out[si * 2 * WT_SIZE + half * WT_SIZE + k_idx * COEFF_LD + col] = sc[i * STENCIL_SIDE + src_col];
            }
          }
        }
      }
    }
  }
}

/** Legacy single-slice builder (calls build_param_matrix<1>). */
inline void build_convstencil_param_matrix_int8(int8_t *param_out, const int8_t *coeffs) {
  build_param_matrix<1>(param_out, coeffs);
}

} // namespace convstencil

using namespace ozi;

/**
 * Computes S @ u using INT8 WMMA tensor core operations (ConvStencil method).
 *
 * Uses the ConvStencil dual-tessellation stencil2row MMA approach for interior
 * elements, with emulated_gain scalar multiplication for boundary elements
 * (matching stencil_v1 boundary semantics).
 *
 * Launch configuration:
 *   Block: convstencil::NUM_WARPS * WARP_SIZE threads (= 128), all in threadIdx.x
 *   Grid:  (ceil(height / TILE_SIZE_ROW), ceil(ldm / TILE_SIZE_COL))
 *
 * @param out       Output slice buffer [n_slices x height x ldm] int8_t
 * @param in        Input slice buffer  [n_slices x height x ldm] int8_t
 * @param weight    Prebuilt weight matrices [M x 2 x WT_SIZE] int8_t
 * @param ldm       Number of columns in the grid (leading dimension)
 * @param height    Number of rows in the grid
 * @param boundary  Boundary gain data (emulated_gain + n_slices) for non-interior elements
 */
template <OzakiPair O, int32_t M = 1>
  requires(std::is_same_v<typename O::target_type, int8_t>) && (M >= 1)
__global__ __launch_bounds__(128, 3) void cvstencil(typename O::target_type *__restrict__ out,
                                                    const typename O::target_type *__restrict__ in,
                                                    const typename O::target_type *__restrict__ weight,
                                                    const size_t ldm, const size_t height,
                                                    OzakiDescriptor<O> *__restrict__ desc,
                                                    __grid_constant__ const convstencil::BoundaryGainData boundary) {
  using namespace convstencil;
  using namespace nvcuda;
  using Target = typename O::target_type;
  using Acc = typename O::acc_type;

  // ======================== Descriptor state ========================
  const int32_t total_slices = desc->n_slices_input;
  const int32_t carry_idx = desc->carry_slice_index;
  const Target threshold = desc->overflow_danger_threshold;
  const int32_t just_shifted = desc->just_shifted;

  // ======================== Thread identity ========================
  const int32_t warp_id = threadIdx.x / WARP_SIZE;
  const int32_t lane_id = threadIdx.x % WARP_SIZE;
  const int32_t tid = threadIdx.x;

  const size_t slice_stride = ldm * height;

  // ======================== Shared memory ========================
  __shared__ __align__(16) int8_t sm_data[2][SM_SIZE_COL * SM_LD];
  __shared__ __align__(16) int32_t sm_staging[NUM_WARPS * ACC_TILE_SIZE];
  __shared__ int32_t sm_overflow[NUM_WARPS];

  // ======================== Initialize overflow flags ========================
  if (tid < NUM_WARPS)
    sm_overflow[tid] = 0;

  // ======================== WMMA working fragments (reused across chunks) ========================
  wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, int8_t, wmma::col_major> in_frag;
  wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, int32_t> acc;

  // Max absolute value of MSB accumulator, for overflow detection
  int32_t max_acc = 0;

  // ======================== Tile coordinates ========================
  // Data tile starts HALO rows/cols BEFORE the output tile to include the stencil neighborhood.
  // This is critical: without this offset, the stencil2row mapping reads wrong input positions.
  const int32_t data_row_start = static_cast<int32_t>(blockIdx.x * TILE_SIZE_ROW) - HALO;
  const int32_t data_col_start = static_cast<int32_t>(blockIdx.y * TILE_SIZE_COL) - HALO;

  // Is the entire data tile within the allocated grid? Used for fast-path loading.
  const bool tile_in_bounds =
      (data_row_start >= 0) && (data_row_start + DATA_TILE_SIZE_ROW <= static_cast<int32_t>(height)) &&
      (data_col_start >= 0) && (data_col_start + DATA_TILE_SIZE_COL <= static_cast<int32_t>(ldm));

  // Helper constants for output element mapping (MMA tile -> thread elements)
  constexpr int32_t LANES_PER_ROW = MMA_N;              // = 8
  constexpr int32_t GROUPS = WARP_SIZE / LANES_PER_ROW; // = 4

  // Zero-initialize sm_data padding (SM_LD - SM_SIZE_ROW = 16 pad bytes per column).
  // Padding is read by WMMA but never written by compute_lut_a/b.
  // Corresponding weight entries are zero, so zero padding is both safe and required.
  for (int32_t idx = tid; idx < 2 * SM_SIZE_COL * SM_LD; idx += blockDim.x)
    reinterpret_cast<int8_t *>(sm_data)[idx] = 0;

  // ====================================================================
  // CHUNKED PROCESSING: outer loop over 2-row chunks, inner over slices
  // Reduces register pressure from M*MMA_N*8 to M*ROWS_PER_CHUNK*8.
  // sm_data is reloaded per (chunk, slice) pair — 4x total loads.
  // ====================================================================
  for (int32_t chunk = 0; chunk < NUM_CHUNKS; ++chunk) {

    // Per-chunk accumulators: only ROWS_PER_CHUNK rows live at once
    int32_t acc_reg[M][ROWS_PER_CHUNK][ELEMS_PER_THREAD];
    int32_t carry_reg[ROWS_PER_CHUNK][ELEMS_PER_THREAD];

#pragma unroll
    for (int32_t si = 0; si < M; ++si)
#pragma unroll
      for (int32_t r = 0; r < ROWS_PER_CHUNK; ++r)
#pragma unroll
        for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e)
          acc_reg[si][r][e] = 0;

#pragma unroll
    for (int32_t r = 0; r < ROWS_PER_CHUNK; ++r)
#pragma unroll
      for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e)
        carry_reg[r][e] = 0;

    // ---- Main data-slice loop for this chunk ----
    for (int32_t logic_s = total_slices - 1; logic_s >= 0; --logic_s) {
      const int32_t phys_idx = (carry_idx + logic_s) % total_slices;
      const bool treat_as_zero = (logic_s == 0 && just_shifted);

      // ---- Phase 1: Load sm_data (all threads cooperate) ----
      if (!treat_as_zero) {
        const Target *in_slice = in + slice_stride * phys_idx;

        if (tile_in_bounds) {
          // Row-by-row fast path with 32-byte sector-aligned loads.
          // data_col_start is typically offset by HALO=3 from a 256-byte tile boundary,
          // landing at byte 29 within a 32-byte sector. Aligning the load pointer down
          // prevents each warp access from spanning two sectors.
          for (int32_t row = 0; row < DATA_TILE_SIZE_ROW; ++row) {
            const Target *row_ptr = in_slice + (data_row_start + row) * ldm + data_col_start;
            const int32_t pad = reinterpret_cast<uintptr_t>(row_ptr) & 31;
            const Target *aligned_ptr = row_ptr - pad;
            const int32_t total = DATA_TILE_SIZE_COL + pad;

            for (int32_t pos = tid; pos < total; pos += blockDim.x) {
              int8_t val = aligned_ptr[pos];
              int32_t col = pos - pad;
              if (col >= 0) {
                sm_data[0][compute_lut_a(row, col)] = val;
                sm_data[1][compute_lut_b(row, col)] = val;
              }
            }
          }
        } else {
          // Row-by-row slow path (bounds-checked)
          for (int32_t row = 0; row < DATA_TILE_SIZE_ROW; ++row) {
            int32_t global_row = data_row_start + row;
            for (int32_t col = tid; col < DATA_TILE_SIZE_COL; col += blockDim.x) {
              int32_t global_col = data_col_start + col;
              int8_t val = 0;
              if (global_row >= 0 && global_row < static_cast<int32_t>(height) && global_col >= 0 &&
                  global_col < static_cast<int32_t>(ldm)) {
                val = in_slice[global_row * ldm + global_col];
              }
              sm_data[0][compute_lut_a(row, col)] = val;
              sm_data[1][compute_lut_b(row, col)] = val;
            }
          }
        }

        if (tid == 0) {
          sm_data[0][TRASH] = 0;
          sm_data[1][TRASH] = 0;
        }
      }
      __syncthreads();

      // ---- Phase 2: Process ROWS_PER_CHUNK rows ----
      for (int32_t ii = 0; ii < ROWS_PER_CHUNK; ++ii) {
        const int32_t i = chunk * ROWS_PER_CHUNK + ii; // global row in warp's stripe

        // Step 2a: Shift sliding window
#pragma unroll
        for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e) {
#pragma unroll
          for (int32_t si = M - 1; si > 0; --si)
            acc_reg[si][ii][e] = acc_reg[si - 1][ii][e];
          acc_reg[0][ii][e] = 0;
          acc_reg[M - 1][ii][e] += carry_reg[ii][e];
          carry_reg[ii][e] = 0;
        }

        // Step 2b: MMA for each stencil slice
        if (!treat_as_zero) {
          const int32_t s_col = (warp_id * MMA_N + i) * STENCIL_SIDE;

          for (int32_t si = 0; si < M; ++si) {
            if (logic_s + si >= total_slices)
              continue;

#pragma unroll
            for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e)
              acc.x[e] = acc_reg[si][ii][e];

#pragma unroll
            for (int32_t half = 0; half < 2; ++half) {
#pragma unroll
              for (int32_t c_idx = 0; c_idx < MMA_NUM; ++c_idx) {
                wmma::load_matrix_sync(in_frag, sm_data[half] + (s_col + c_idx * MMA_K) * SM_LD, SM_LD);
                wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, int8_t, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag,
                                       weight + si * 2 * WT_SIZE + half * WT_SIZE + c_idx * MMA_K * COEFF_LD,
                                       COEFF_LD);
                wmma::mma_sync(acc, in_frag, b_frag, acc);
              }
            }

#pragma unroll
            for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e)
              acc_reg[si][ii][e] = acc.x[e];
          }
        }

        // Step 2c: Extract output
        const int32_t out_pos = logic_s + M - 1;
        if (out_pos >= 0 && out_pos < total_slices) {
          const int32_t out_phys = (carry_idx + out_pos) % total_slices;
          Target *slice_out = out + slice_stride * out_phys;

#pragma unroll
          for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e)
            acc.x[e] = acc_reg[M - 1][ii][e];

          int32_t *warp_staging = sm_staging + warp_id * ACC_TILE_SIZE;
          wmma::store_matrix_sync(warp_staging, acc, MMA_N, wmma::mem_row_major);

          if (out_pos != 0) {
#pragma unroll
            for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e) {
              int32_t val = acc_reg[M - 1][ii][e];
              Target trunc = static_cast<Target>(val);
              carry_reg[ii][e] = (val - static_cast<Acc>(trunc)) >> (CHAR_BIT * sizeof(Target) - 1);
            }
          }

          const int32_t out_row_local = warp_id * MMA_N + i;
          const int32_t global_row = static_cast<int32_t>(blockIdx.x * TILE_SIZE_ROW) + out_row_local;

#pragma unroll
          for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e) {
            int32_t local_col = lane_id % LANES_PER_ROW;
            int32_t local_row = (lane_id / LANES_PER_ROW) + e * GROUPS;
            int32_t flat_idx = local_row * MMA_N + local_col;
            int32_t out_col = local_row * LANES_PER_ROW + local_col;
            int32_t global_col = static_cast<int32_t>(blockIdx.y * TILE_SIZE_COL) + out_col;

            bool in_grid = (global_row < static_cast<int32_t>(height) && global_col < static_cast<int32_t>(ldm));
            bool is_interior = in_grid && (global_row >= HALO && global_row < static_cast<int32_t>(height) - HALO &&
                                           global_col >= HALO && global_col < static_cast<int32_t>(ldm) - HALO);

            if (is_interior) {
              slice_out[global_row * ldm + global_col] = static_cast<Target>(warp_staging[flat_idx]);
            }

            if (out_pos == 0 && is_interior) {
              max_acc = max(max_acc, abs(acc_reg[M - 1][ii][e]));
            }
          }
        }
      } // end ROWS_PER_CHUNK loop

      __syncthreads();
    } // end data-slice loop

    // ---- Cleanup for this chunk: flush M-1 remaining positions ----
    for (int32_t p = M - 2; p >= 0; --p) {
      for (int32_t ii = 0; ii < ROWS_PER_CHUNK; ++ii) {
        const int32_t i = chunk * ROWS_PER_CHUNK + ii;

#pragma unroll
        for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e) {
#pragma unroll
          for (int32_t si = M - 1; si > 0; --si)
            acc_reg[si][ii][e] = acc_reg[si - 1][ii][e];
          acc_reg[0][ii][e] = 0;
          acc_reg[M - 1][ii][e] += carry_reg[ii][e];
          carry_reg[ii][e] = 0;
        }

        if (p < total_slices) {
          const int32_t out_phys = (carry_idx + p) % total_slices;
          Target *slice_out = out + slice_stride * out_phys;

#pragma unroll
          for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e)
            acc.x[e] = acc_reg[M - 1][ii][e];

          int32_t *warp_staging = sm_staging + warp_id * ACC_TILE_SIZE;
          wmma::store_matrix_sync(warp_staging, acc, MMA_N, wmma::mem_row_major);

          if (p != 0) {
#pragma unroll
            for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e) {
              int32_t val = acc_reg[M - 1][ii][e];
              Target trunc = static_cast<Target>(val);
              carry_reg[ii][e] = (val - static_cast<Acc>(trunc)) >> (CHAR_BIT * sizeof(Target) - 1);
            }
          }

          const int32_t out_row_local = warp_id * MMA_N + i;
          const int32_t global_row = static_cast<int32_t>(blockIdx.x * TILE_SIZE_ROW) + out_row_local;

#pragma unroll
          for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e) {
            int32_t local_col = lane_id % LANES_PER_ROW;
            int32_t local_row = (lane_id / LANES_PER_ROW) + e * GROUPS;
            int32_t flat_idx = local_row * MMA_N + local_col;
            int32_t out_col = local_row * LANES_PER_ROW + local_col;
            int32_t global_col = static_cast<int32_t>(blockIdx.y * TILE_SIZE_COL) + out_col;

            bool in_grid = (global_row < static_cast<int32_t>(height) && global_col < static_cast<int32_t>(ldm));
            bool is_interior = in_grid && (global_row >= HALO && global_row < static_cast<int32_t>(height) - HALO &&
                                           global_col >= HALO && global_col < static_cast<int32_t>(ldm) - HALO);

            if (is_interior) {
              slice_out[global_row * ldm + global_col] = static_cast<Target>(warp_staging[flat_idx]);
            }

            if (p == 0 && is_interior) {
              max_acc = max(max_acc, abs(acc_reg[M - 1][ii][e]));
            }
          }
        }
      } // end ROWS_PER_CHUNK cleanup inner
    } // end cleanup position loop

  } // end chunk loop

  // ====================================================================
  // BOUNDARY HANDLING: emulated_gain for non-interior elements
  // Boundary elements whose stencil window extends beyond the domain cannot
  // use the MMA result (it was computed from zero-padded or partial data).
  // Instead, apply the emulated_gain scalar multiplication: a full NxN_bg
  // digit multiplication of data slices against the boundary gain digits,
  // producing results for all output slices at once.
  // ====================================================================
  if (boundary.n_slices > 0) {
    for (int32_t i = 0; i < MMA_N; ++i) {
      int32_t out_row_local = warp_id * MMA_N + i;
      int32_t global_row = static_cast<int32_t>(blockIdx.x * TILE_SIZE_ROW) + out_row_local;

      for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e) {
        int32_t local_col = lane_id % LANES_PER_ROW;
        int32_t local_row = (lane_id / LANES_PER_ROW) + e * GROUPS;
        int32_t out_col = local_row * LANES_PER_ROW + local_col;
        int32_t global_col = static_cast<int32_t>(blockIdx.y * TILE_SIZE_COL) + out_col;

        bool in_grid = (global_row < static_cast<int32_t>(height) && global_col < static_cast<int32_t>(ldm));
        bool is_interior = in_grid && (global_row >= HALO && global_row < static_cast<int32_t>(height) - HALO &&
                                       global_col >= HALO && global_col < static_cast<int32_t>(ldm) - HALO);

        if (in_grid && !is_interior) {
          size_t global_offset = static_cast<size_t>(global_row) * ldm + global_col;

          // Local result array for the NxN_bg multiplication.
          // Indexed by result position [0..total_slices-1].
          int32_t bnd_result[MAX_DATA_SLICES];
          for (int32_t r = 0; r < total_slices; ++r)
            bnd_result[r] = 0;

          // Accumulate products: data[d] x gain[g] -> result[d + g]
          for (int32_t d = total_slices - 1; d >= 0; --d) {
            if (d == 0 && just_shifted)
              continue;
            int32_t phys_d = (carry_idx + d) % total_slices;
            int32_t data_d = static_cast<int32_t>(in[slice_stride * phys_d + global_offset]);

            for (int32_t g = 0; g < boundary.n_slices; ++g) {
              int32_t r = d + g;
              if (r < total_slices)
                bnd_result[r] += data_d * static_cast<int32_t>(boundary.emulated_gain[g]);
            }
          }

          // Carry propagation (LSB -> MSB) and write all output slices
          for (int32_t r = total_slices - 1; r > 0; --r) {
            int32_t phys_r = (carry_idx + r) % total_slices;
            Target digit = static_cast<Target>(bnd_result[r]);
            out[slice_stride * phys_r + global_offset] = digit;
            bnd_result[r - 1] += (bnd_result[r] - static_cast<int32_t>(digit)) >> (CHAR_BIT * sizeof(Target) - 1);
          }
          // MSB: no carry target
          int32_t phys_0 = carry_idx % total_slices;
          out[slice_stride * phys_0 + global_offset] = static_cast<Target>(bnd_result[0]);

          // Overflow tracking for boundary element (using MSB value)
          max_acc = max(max_acc, abs(bnd_result[0]));
        }
      }
    }
  }

  // ====================================================================
  // OVERFLOW DETECTION (matches stencil_v1 pattern)
  // ====================================================================
  __syncthreads();
  int32_t dangerous = __any_sync(__activemask(), max_acc > static_cast<Acc>(threshold));
  if (lane_id == 0)
    sm_overflow[warp_id] = dangerous;
  __syncthreads();

  if (tid < WARP_SIZE) {
    int32_t flag = (tid < NUM_WARPS) ? sm_overflow[tid] : 0;
    flag = __any_sync(__activemask(), flag);
    if (flag && tid == 0) {
      atomicExch(&desc->shift_required, 1);
    }
  }
}
} // namespace ozigpu

#endif // HEADER_OZI_STENCIL_CONV
