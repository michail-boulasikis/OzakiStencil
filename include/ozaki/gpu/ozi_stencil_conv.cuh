#ifndef HEADER_OZI_STENCIL_CONV
#define HEADER_OZI_STENCIL_CONV

#include "../core/oztraits.hpp"
#include "ozi.cuh"
#include "ozi_kernels.cuh"

#include <climits>
#include <concepts>
#include <cstring>
#include <cuda_runtime.h>
#include <type_traits>

namespace ozigpu {
namespace convstencil {

// ========== Architecture constants (hardcoded for 7x7 stencil) ==========
constexpr int32_t MMA_M = 16;
constexpr int32_t MMA_N = 8;
constexpr int32_t MMA_K = 16;

constexpr int32_t HALO = 3;
constexpr int32_t STENCIL_SIDE = 7;
constexpr int32_t STRIDE = STENCIL_SIDE + 1; // = 8 (power of 2 for 7x7)

constexpr int32_t NUM_WARPS = 4;
constexpr int32_t MMA_NUM = (STENCIL_SIDE * STENCIL_SIDE + MMA_K - 1) / MMA_K; // = 4

// Tile dimensions: output tile = TILE_SIZE_ROW x TILE_SIZE_COL
// Data tile = output tile + 2 x HALO in each direction
constexpr int32_t TILE_SIZE_ROW = NUM_WARPS * MMA_N;             // = 32
constexpr int32_t TILE_SIZE_COL = MMA_M * MMA_N;                 // = 128
constexpr int32_t DATA_TILE_SIZE_ROW = TILE_SIZE_ROW + 2 * HALO; // = 38
constexpr int32_t DATA_TILE_SIZE_COL = TILE_SIZE_COL + 2 * HALO; // = 134

// Stencil2row shared memory dimensions (row-major: K is fast dim, XOR-swizzled)
// SMEM_ROWS = number of column groups = floor(DATA_TILE_SIZE_COL / STRIDE) = MMA_M = 16
// SMEM_K_EXTENT = K-extent covering all output rows' stencil windows + MMA tile padding
// K_STRIDE = physical row stride in bytes (must be a multiple of 4 and K_STRIDE/4 % 32 != 0 for swizzle)
constexpr int32_t SMEM_ROWS = DATA_TILE_SIZE_COL / STRIDE;                        // = 16
constexpr int32_t K_STRIDE = 352;                                                 // 352/4=88, 88%32=24
constexpr int32_t SMEM_K_EXTENT = (TILE_SIZE_ROW - 1) * STRIDE + MMA_NUM * MMA_K; // = 312

constexpr int32_t COEFF_LD = MMA_N;                      // = 8 (weight matrix leading dim)
constexpr int32_t WT_SIZE = MMA_NUM * MMA_K * COEFF_LD;  // = 512 (per-tessellation weight size in bytes)
constexpr int32_t PACKED_WT_WORDS = MMA_NUM * WARP_SIZE; // = 128 (uint32 words per tessellation per slice)
static_assert(PACKED_WT_WORDS * 4 == WT_SIZE, "Packed weight size must match WT_SIZE in bytes");
constexpr int32_t ACC_TILE_SIZE = MMA_M * MMA_N;                // = 128 (elements per MMA tile)
constexpr int32_t ELEMS_PER_THREAD = ACC_TILE_SIZE / WARP_SIZE; // = 4
constexpr int32_t MAX_M = 12;                                   // max supported stencil slices
constexpr int32_t MAX_DATA_SLICES = 24;                         // matches StencilSlices::MAX_DATA_SLICES

// Shared memory sink index: invalid tessellation positions write here instead of
// corrupting real data. Sits at the last byte of the sm_data allocation, which is
// never read by the MMA. Zeroed before each slice iteration.
constexpr int32_t SMEM_SINK = SMEM_ROWS * K_STRIDE - 1;

// Chunking constants: the 8 output rows (MMA_N) per warp are split into chunks.
// One chunk's accumulator state lives in registers; the rest are swapped in/out
// of local memory (compiler-managed spills) via rotation.
constexpr int32_t ROWS_PER_CHUNK = 2;
constexpr int32_t NUM_CHUNKS = MMA_N / ROWS_PER_CHUNK; // = 4
static_assert(MMA_N % ROWS_PER_CHUNK == 0, "MMA_N must be divisible by ROWS_PER_CHUNK");

// ========== Boundary gain data (passed by value to kernel) ==========
struct BoundaryGainData {
  int8_t emulated_gain[MAX_DATA_SLICES];
  int32_t n_slices;
};

// ========== Shared memory index computation ==========
// Compute the swizzled sm_data index for a data element at (data_row, data_col)
// within the data tile. Uses row-major layout with Swizzle<3,0,0> (word-level XOR).
// Returns SMEM_SINK for positions outside the valid tessellation range.
// Valid only for STRIDE = 8 (power of 2); other stencil sizes need integer division.

/** Upper tessellation (sm_data[0]): each STRIDE-wide column group maps the upper
 *  portion of the stencil's column overlap pattern. */
__device__ __forceinline__ int32_t smem_index_upper(int32_t data_row, int32_t data_col) {
  bool valid = (((data_col + 1) & (STRIDE - 1)) != 0) && (data_col < DATA_TILE_SIZE_COL - 2 * HALO - 1);
  int32_t sm_row = data_col >> 3;
  int32_t sm_col = STRIDE * data_row + (data_col & (STRIDE - 1));
  int32_t word = sm_col >> 2;
  int32_t byte_in_word = sm_col & 3;
  int32_t swizzled_word = word ^ (sm_row & 7);
  return valid ? (sm_row * K_STRIDE + swizzled_word * 4 + byte_in_word) : SMEM_SINK;
}

/** Lower tessellation (sm_data[1]): each STRIDE-wide column group maps the lower
 *  portion of the stencil's column overlap pattern, shifted by STENCIL_SIDE. */
__device__ __forceinline__ int32_t smem_index_lower(int32_t data_row, int32_t data_col) {
  bool valid = (((data_col + 2) & (STRIDE - 1)) != 0) && (data_col > 2 * HALO);
  int32_t col_shifted = data_col - STENCIL_SIDE;
  int32_t sm_row = col_shifted >> 3;
  int32_t sm_col = STRIDE * data_row + (col_shifted & (STRIDE - 1));
  int32_t word = sm_col >> 2;
  int32_t byte_in_word = sm_col & 3;
  int32_t swizzled_word = word ^ (sm_row & 7);
  return valid ? (sm_row * K_STRIDE + swizzled_word * 4 + byte_in_word) : SMEM_SINK;
}

// ========== Weight matrix builder for M stencil slices ==========

/** Build the dual-tessellation weight matrices for M stencil slices.
 *
 *  For each stencil slice and tessellation (upper/lower), builds a K x N weight
 *  matrix encoding the stencil's column overlap pattern, then packs it into a
 *  lane-indexed uint32 layout matching the m16n8k16 B-fragment mapping for
 *  coalesced shared memory loads.
 *
 *  @param param_out  Output buffer: [M * 2 * WT_SIZE] bytes, zero-initialized internally.
 *  @param coeffs     Flat input: [M * STENCIL_SIDE^2] int8_t, slice 0 (MSB) first,
 *                    each slice in row-major stencil order.
 */
template <int32_t M> inline void build_param_matrix(int8_t *param_out, const int8_t *coeffs) {
  constexpr int32_t N_COEFFS = STENCIL_SIDE * STENCIL_SIDE;
  uint32_t *packed_out = reinterpret_cast<uint32_t *>(param_out);
  std::memset(packed_out, 0, M * 2 * WT_SIZE);

  int8_t weight_block[WT_SIZE]; // scratch for one (slice, tessellation) block in K x N layout

  for (int32_t slice = 0; slice < M; ++slice) {
    const int8_t *slice_coeffs = coeffs + slice * N_COEFFS;

    for (int32_t tess = 0; tess < 2; ++tess) {
      std::memset(weight_block, 0, WT_SIZE);

      // Populate the K x N weight block from stencil coefficients.
      // Upper tessellation (tess=0): stencil column j contributes to output columns 0..j
      // Lower tessellation (tess=1): stencil column j contributes to output columns (j+1)..6
      for (int32_t stencil_row = 0; stencil_row < STENCIL_SIDE; ++stencil_row) {
        for (int32_t stencil_col = 0; stencil_col < STENCIL_SIDE; ++stencil_col) {
          int32_t k_idx = stencil_row * STRIDE + stencil_col;

          for (int32_t n_col = 0; n_col < COEFF_LD; ++n_col) {
            int32_t coeff_col;
            bool valid;
            if (tess == 0) {
              valid = (stencil_col >= n_col);
              coeff_col = stencil_col - n_col;
            } else {
              valid = (stencil_col < n_col);
              coeff_col = stencil_col - n_col + STENCIL_SIDE;
            }
            // When valid is true, coeff_col is always in [0, STENCIL_SIDE) -- guaranteed
            // by the tessellation geometry (upper: j >= n_col implies j - n_col >= 0;
            // lower: j < n_col implies j - n_col + 7 in [0, 6]).
            if (valid)
              weight_block[k_idx * COEFF_LD + n_col] = slice_coeffs[stencil_row * STENCIL_SIDE + coeff_col];
          }
        }
      }

      // Pack into lane-indexed uint32 layout for coalesced B-fragment loads.
      // Each uint32 holds 4 consecutive K-dimension bytes for one lane.
      uint32_t *dst = packed_out + (slice * 2 + tess) * PACKED_WT_WORDS;
      for (int32_t k_tile = 0; k_tile < MMA_NUM; ++k_tile) {
        const int8_t *tile_ptr = weight_block + k_tile * MMA_K * COEFF_LD;
        for (int32_t lane = 0; lane < WARP_SIZE; ++lane) {
          int32_t groupID = lane >> 2;
          int32_t tIG = lane & 3;
          dst[k_tile * WARP_SIZE + lane] =
              static_cast<uint32_t>(static_cast<uint8_t>(tile_ptr[(tIG * 4 + 0) * COEFF_LD + groupID])) |
              static_cast<uint32_t>(static_cast<uint8_t>(tile_ptr[(tIG * 4 + 1) * COEFF_LD + groupID])) << 8 |
              static_cast<uint32_t>(static_cast<uint8_t>(tile_ptr[(tIG * 4 + 2) * COEFF_LD + groupID])) << 16 |
              static_cast<uint32_t>(static_cast<uint8_t>(tile_ptr[(tIG * 4 + 3) * COEFF_LD + groupID])) << 24;
        }
      }
    }
  }
}

// ========== Inline PTX helpers ==========

/** Load A-fragment (m16n8k16) from swizzled row-major sm_data using 2x LDS.32.
 *  Zero bank conflicts with Swizzle<3,0,0> and K_STRIDE where K_STRIDE/4 % 32 == 8.
 *
 *  @param A         Output: 2 registers (one per m8-row group)
 *  @param sm_base   Base pointer to sm_data[tessellation]
 *  @param k_start   Starting K-byte offset (must be a multiple of 4)
 *  @param lane_id   Thread lane within warp
 */
__device__ __forceinline__ void load_a_frag_swizzled(uint32_t (&A)[2], const int8_t *sm_base, int32_t k_start,
                                                     int32_t lane_id) {
  const int32_t groupID = lane_id >> 2;
  const int32_t tIG = lane_id & 3;
  const int32_t base_word = (k_start >> 2) + tIG;
#pragma unroll
  for (int32_t r = 0; r < 2; ++r) {
    int32_t m = groupID + r * 8;
    int32_t swizzled_word = base_word ^ (m & 7);
    A[r] = *reinterpret_cast<const uint32_t *>(sm_base + m * K_STRIDE + swizzled_word * 4);
  }
}

/** Compute the output column for accumulator element `elem` held by the given lane.
 *  Based on the m16n8k16 accumulator fragment mapping:
 *    MMA tile row: groupID + 8 * (elem >> 1)
 *    MMA tile col: tIG * 2 + (elem & 1)
 *    Output column: row * MMA_N + col
 */
__device__ __forceinline__ int32_t acc_out_col(int32_t elem, int32_t lane_id) {
  const int32_t groupID = lane_id >> 2;
  const int32_t tIG = lane_id & 3;
  return (groupID + 8 * (elem >> 1)) * MMA_N + tIG * 2 + (elem & 1);
}

/** Advance the accumulator sliding window by one position.
 *  Shifts acc[0..M-2] up to acc[1..M-1], clears acc[0] for the next MMA result,
 *  then injects the previous carry into the MSB slot (acc[M-1]) and clears carry. */
template <int32_t M>
__device__ __forceinline__ void advance_sliding_window(int32_t (&acc)[M][ROWS_PER_CHUNK][ELEMS_PER_THREAD],
                                                       int32_t (&carry)[ROWS_PER_CHUNK][ELEMS_PER_THREAD],
                                                       int32_t row_in_chunk) {
#pragma unroll
  for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem) {
#pragma unroll
    for (int32_t ws = M - 1; ws > 0; --ws)
      acc[ws][row_in_chunk][elem] = acc[ws - 1][row_in_chunk][elem];
    acc[0][row_in_chunk][elem] = 0;
    acc[M - 1][row_in_chunk][elem] += carry[row_in_chunk][elem];
    carry[row_in_chunk][elem] = 0;
  }
}

/** Extract the truncation carry from the MSB accumulator slot.
 *  Truncates acc[M-1] to Target (int8) and computes the carry as the
 *  sign-extended high bits lost in truncation: (val - int8(val)) >> 7. */
template <typename Target, typename Acc, int32_t M>
__device__ __forceinline__ void extract_truncation_carry(int32_t (&acc)[M][ROWS_PER_CHUNK][ELEMS_PER_THREAD],
                                                         int32_t (&carry)[ROWS_PER_CHUNK][ELEMS_PER_THREAD],
                                                         int32_t row_in_chunk) {
#pragma unroll
  for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem) {
    int32_t val = acc[M - 1][row_in_chunk][elem];
    Target trunc = static_cast<Target>(val);
    carry[row_in_chunk][elem] = (val - static_cast<Acc>(trunc)) >> (CHAR_BIT * sizeof(Target) - 1);
  }
}

} // namespace convstencil

using namespace ozi;

/**
 * Computes S @ u using INT8 tensor core operations via inline PTX (ConvStencil method).
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
__global__ __launch_bounds__(128) void cvstencil(typename O::target_type *__restrict__ out,
                                                 const typename O::target_type *__restrict__ in,
                                                 const typename O::target_type *__restrict__ weight, const size_t ldm,
                                                 const size_t height, OzakiDescriptor<O> *__restrict__ desc,
                                                 __grid_constant__ const convstencil::BoundaryGainData boundary) {
  asm volatile(".pragma \"enable_smem_spilling\";");
  using namespace convstencil;
  using Target = typename O::target_type;
  using Acc = typename O::acc_type;

  // ======================== Descriptor state ========================
  const int32_t total_slices = desc->n_slices_input;
  const int32_t carry_idx = desc->carry_slice_index;
  const Target threshold = desc->overflow_danger_threshold;
  const int32_t just_shifted = desc->just_shifted;
  const int32_t *phys_lut = &phys_lut_cmem[carry_idx * PHYS_LUT_MAX_SLICES];

  // ======================== Thread identity ========================
  const int32_t warp_id = threadIdx.x / WARP_SIZE;
  const int32_t lane_id = threadIdx.x % WARP_SIZE;
  const int32_t tid = threadIdx.x;

  const size_t slice_stride = ldm * height;

  // ======================== Shared memory ========================
  __shared__ __align__(16) int8_t sm_data[2][SMEM_ROWS * K_STRIDE];
  __shared__ __align__(16) uint32_t sm_weight[M * 2 * PACKED_WT_WORDS];
  __shared__ int32_t sm_overflow[NUM_WARPS];

  if (tid < NUM_WARPS)
    sm_overflow[tid] = 0;

  int32_t max_abs_acc = 0; // tracks max |accumulator| for overflow detection

  // ======================== Tile coordinates ========================
  // Data tile starts HALO rows/cols BEFORE the output tile to include the stencil neighborhood.
  const int32_t data_row_start = static_cast<int32_t>(blockIdx.x * TILE_SIZE_ROW) - HALO;
  const int32_t data_col_start = static_cast<int32_t>(blockIdx.y * TILE_SIZE_COL) - HALO;

  const bool tile_in_bounds =
      (data_row_start >= 0) && (data_row_start + DATA_TILE_SIZE_ROW <= static_cast<int32_t>(height)) &&
      (data_col_start >= 0) && (data_col_start + DATA_TILE_SIZE_COL <= static_cast<int32_t>(ldm));

  // MMA accumulator-to-output mapping constants (used in boundary handling)
  constexpr int32_t LANES_PER_ROW = MMA_N;              // = 8
  constexpr int32_t GROUPS = WARP_SIZE / LANES_PER_ROW; // = 4

  // Zero-initialize sm_data. Padded regions (K_STRIDE gap, STRIDE-1 padding) are
  // read by MMA but never written by smem_index_upper/lower; the matching weight
  // entries are zero, so zero padding is both safe and required.
  static_assert(2 * SMEM_ROWS * K_STRIDE % 4 == 0);
  {
    int32_t *sm_flat = reinterpret_cast<int32_t *>(sm_data);
    constexpr int32_t SM_WORDS = 2 * SMEM_ROWS * K_STRIDE / 4;
    for (int32_t idx = tid; idx < SM_WORDS; idx += blockDim.x)
      sm_flat[idx] = 0;
  }

  // Load weight matrices from global to shared memory (eliminates repeated
  // global reads during the MMA inner loop).
  {
    const uint32_t *wt_src = reinterpret_cast<const uint32_t *>(weight);
    constexpr int32_t WT_WORDS = M * 2 * PACKED_WT_WORDS;
    for (int32_t idx = tid; idx < WT_WORDS; idx += blockDim.x)
      sm_weight[idx] = wt_src[idx];
  }

  // ====================================================================
  // ROTATED CHUNK PROCESSING
  //
  // The 8 output rows per warp are split into NUM_CHUNKS chunks of
  // ROWS_PER_CHUNK rows. sm_data is loaded once per data slice (outer loop).
  // For each slice, all chunks are processed via rotation one at a time.
  // ====================================================================

  // Active chunk state (persists in registers)
  int32_t acc_reg[M][ROWS_PER_CHUNK][ELEMS_PER_THREAD];
  int32_t carry_reg[ROWS_PER_CHUNK][ELEMS_PER_THREAD];

  // Inactive chunk state (in local memory, one buffer per inactive chunk)
  int32_t saved_acc[NUM_CHUNKS - 1][M][ROWS_PER_CHUNK][ELEMS_PER_THREAD];
  int32_t saved_carry[NUM_CHUNKS - 1][ROWS_PER_CHUNK][ELEMS_PER_THREAD];

  int32_t active_chunk = 0;

  // Zero-initialize all chunk state
#pragma unroll
  for (int32_t ws = 0; ws < M; ++ws)
#pragma unroll
    for (int32_t r = 0; r < ROWS_PER_CHUNK; ++r)
#pragma unroll
      for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem)
        acc_reg[ws][r][elem] = 0;

#pragma unroll
  for (int32_t r = 0; r < ROWS_PER_CHUNK; ++r)
#pragma unroll
    for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem)
      carry_reg[r][elem] = 0;

#pragma unroll
  for (int32_t buf = 0; buf < NUM_CHUNKS - 1; ++buf)
#pragma unroll
    for (int32_t ws = 0; ws < M; ++ws)
#pragma unroll
      for (int32_t r = 0; r < ROWS_PER_CHUNK; ++r)
#pragma unroll
        for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem)
          saved_acc[buf][ws][r][elem] = 0;

#pragma unroll
  for (int32_t buf = 0; buf < NUM_CHUNKS - 1; ++buf)
#pragma unroll
    for (int32_t r = 0; r < ROWS_PER_CHUNK; ++r)
#pragma unroll
      for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem)
        saved_carry[buf][r][elem] = 0;

  // ---- Main data-slice loop (outer: load sm_data once per slice) ----
  for (int32_t slice_idx = total_slices - 1; slice_idx >= 0; --slice_idx) {
    const int32_t physical_slice = phys_lut[slice_idx];
    const bool skip_slice = (slice_idx == 0 && just_shifted);

    // Load input data into shared memory (stencil2row transform)
    if (!skip_slice) {
      const Target *in_slice = in + slice_stride * physical_slice;

      if (tile_in_bounds) {
        // Fast path: sector-aligned loads, no bounds checking.
        // data_col_start is offset by HALO=3 from a tile boundary; aligning
        // the load pointer down to a 32-byte boundary avoids split sectors.
        for (int32_t row = 0; row < DATA_TILE_SIZE_ROW; ++row) {
          const Target *row_ptr = in_slice + (data_row_start + row) * ldm + data_col_start;
          const int32_t pad = reinterpret_cast<uintptr_t>(row_ptr) & 31;
          const Target *aligned_ptr = row_ptr - pad;
          const int32_t total = DATA_TILE_SIZE_COL + pad;

          for (int32_t pos = tid; pos < total; pos += blockDim.x) {
            int8_t val = aligned_ptr[pos];
            int32_t col = pos - pad;
            if (col >= 0) {
              sm_data[0][smem_index_upper(row, col)] = val;
              sm_data[1][smem_index_lower(row, col)] = val;
            }
          }
        }
      } else {
        // Slow path: bounds-checked element-by-element
        for (int32_t row = 0; row < DATA_TILE_SIZE_ROW; ++row) {
          int32_t global_row = data_row_start + row;
          for (int32_t col = tid; col < DATA_TILE_SIZE_COL; col += blockDim.x) {
            int32_t global_col = data_col_start + col;
            int8_t val = 0;
            if (global_row >= 0 && global_row < static_cast<int32_t>(height) && global_col >= 0 &&
                global_col < static_cast<int32_t>(ldm)) {
              val = in_slice[global_row * ldm + global_col];
            }
            sm_data[0][smem_index_upper(row, col)] = val;
            sm_data[1][smem_index_lower(row, col)] = val;
          }
        }
      }

      if (tid == 0) {
        sm_data[0][SMEM_SINK] = 0;
        sm_data[1][SMEM_SINK] = 0;
      }
    }
    __syncthreads();

    // Process all chunks via rotation
    for (int32_t pass = 0; pass < NUM_CHUNKS; ++pass) {
      const int32_t chunk = (active_chunk + pass) % NUM_CHUNKS;

      for (int32_t row_in_chunk = 0; row_in_chunk < ROWS_PER_CHUNK; ++row_in_chunk) {
        const int32_t tile_row = chunk * ROWS_PER_CHUNK + row_in_chunk;

        // Advance sliding window: shift accumulators and inject carry
        advance_sliding_window<M>(acc_reg, carry_reg, row_in_chunk);

        // Tensor core matrix multiply (inline PTX m16n8k16)
        if (!skip_slice) {
          const int32_t k_base = (warp_id * MMA_N + tile_row) * STRIDE;

          for (int32_t ws = 0; ws < M; ++ws) {
            if (slice_idx + ws >= total_slices)
              continue;

            uint32_t mma_accum[ELEMS_PER_THREAD];
#pragma unroll
            for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem)
              mma_accum[elem] = static_cast<uint32_t>(acc_reg[ws][row_in_chunk][elem]);

#pragma unroll
            for (int32_t tess = 0; tess < 2; ++tess) {
#pragma unroll
              for (int32_t k_tile = 0; k_tile < MMA_NUM; ++k_tile) {
                uint32_t A[2];
                load_a_frag_swizzled(A, sm_data[tess], k_base + k_tile * MMA_K, lane_id);
                uint32_t B = sm_weight[(ws * 2 + tess) * PACKED_WT_WORDS + k_tile * WARP_SIZE + lane_id];

                asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                             "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                             : "+r"(mma_accum[0]), "+r"(mma_accum[1]), "+r"(mma_accum[2]), "+r"(mma_accum[3])
                             : "r"(A[0]), "r"(A[1]), "r"(B));
              }
            }

#pragma unroll
            for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem)
              acc_reg[ws][row_in_chunk][elem] = static_cast<int32_t>(mma_accum[elem]);
          }
        }

        // Write output and extract carry
        const int32_t output_slice = slice_idx + M - 1;
        if (output_slice >= 0 && output_slice < total_slices) {
          const int32_t output_physical = phys_lut[output_slice];
          Target *slice_out = out + slice_stride * output_physical;

          if (output_slice != 0)
            extract_truncation_carry<Target, Acc, M>(acc_reg, carry_reg, row_in_chunk);

          const int32_t out_row_local = warp_id * MMA_N + tile_row;
          const int32_t global_row = static_cast<int32_t>(blockIdx.x * TILE_SIZE_ROW) + out_row_local;

          if (tile_in_bounds) {
            // Vectorized 16-bit stores: element pairs (0,1) and (2,3) are adjacent columns
#pragma unroll
            for (int32_t store_pair = 0; store_pair < ELEMS_PER_THREAD / 2; ++store_pair) {
              const int32_t elem_lo = store_pair * 2;
              const int32_t out_col = acc_out_col(elem_lo, lane_id);
              const int32_t global_col = static_cast<int32_t>(blockIdx.y * TILE_SIZE_COL) + out_col;
              const uint8_t v0 = static_cast<uint8_t>(static_cast<Target>(acc_reg[M - 1][row_in_chunk][elem_lo]));
              const uint8_t v1 = static_cast<uint8_t>(static_cast<Target>(acc_reg[M - 1][row_in_chunk][elem_lo + 1]));
              *reinterpret_cast<uint16_t *>(&slice_out[global_row * ldm + global_col]) =
                  static_cast<uint16_t>(v0) | (static_cast<uint16_t>(v1) << 8);
            }
            if (output_slice == 0) {
#pragma unroll
              for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem)
                max_abs_acc = max(max_abs_acc, abs(acc_reg[M - 1][row_in_chunk][elem]));
            }
          } else {
#pragma unroll
            for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem) {
              int32_t out_col = acc_out_col(elem, lane_id);
              int32_t global_col = static_cast<int32_t>(blockIdx.y * TILE_SIZE_COL) + out_col;

              bool in_grid = (global_row < static_cast<int32_t>(height) && global_col < static_cast<int32_t>(ldm));
              bool is_interior = in_grid && (global_row >= HALO && global_row < static_cast<int32_t>(height) - HALO &&
                                             global_col >= HALO && global_col < static_cast<int32_t>(ldm) - HALO);

              if (is_interior) {
                slice_out[global_row * ldm + global_col] = static_cast<Target>(acc_reg[M - 1][row_in_chunk][elem]);
              }

              if (output_slice == 0 && is_interior) {
                max_abs_acc = max(max_abs_acc, abs(acc_reg[M - 1][row_in_chunk][elem]));
              }
            }
          }
        }
      } // end row_in_chunk loop

      // Swap active chunk with next saved chunk (rotation)
      if (pass < NUM_CHUNKS - 1) {
#pragma unroll
        for (int32_t ws = 0; ws < M; ++ws)
#pragma unroll
          for (int32_t r = 0; r < ROWS_PER_CHUNK; ++r)
#pragma unroll
            for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem) {
              int32_t tmp = saved_acc[pass][ws][r][elem];
              saved_acc[pass][ws][r][elem] = acc_reg[ws][r][elem];
              acc_reg[ws][r][elem] = tmp;
            }
#pragma unroll
        for (int32_t r = 0; r < ROWS_PER_CHUNK; ++r)
#pragma unroll
          for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem) {
            int32_t tmp = saved_carry[pass][r][elem];
            saved_carry[pass][r][elem] = carry_reg[r][elem];
            carry_reg[r][elem] = tmp;
          }
      }
    } // end pass loop

    active_chunk = (active_chunk + NUM_CHUNKS - 1) % NUM_CHUNKS;

    __syncthreads(); // protect sm_data for next slice iteration
  } // end slice loop

  // ---- Cleanup: flush M-1 remaining output positions for all chunks ----
  for (int32_t cleanup_pass = 0; cleanup_pass < NUM_CHUNKS; ++cleanup_pass) {
    const int32_t chunk = (active_chunk + cleanup_pass) % NUM_CHUNKS;

    for (int32_t flush_pos = M - 2; flush_pos >= 0; --flush_pos) {
      for (int32_t row_in_chunk = 0; row_in_chunk < ROWS_PER_CHUNK; ++row_in_chunk) {
        const int32_t tile_row = chunk * ROWS_PER_CHUNK + row_in_chunk;

        advance_sliding_window<M>(acc_reg, carry_reg, row_in_chunk);

        if (flush_pos < total_slices) {
          const int32_t output_physical = phys_lut[flush_pos];
          Target *slice_out = out + slice_stride * output_physical;

          if (flush_pos != 0)
            extract_truncation_carry<Target, Acc, M>(acc_reg, carry_reg, row_in_chunk);

          const int32_t out_row_local = warp_id * MMA_N + tile_row;
          const int32_t global_row = static_cast<int32_t>(blockIdx.x * TILE_SIZE_ROW) + out_row_local;

          if (tile_in_bounds) {
#pragma unroll
            for (int32_t store_pair = 0; store_pair < ELEMS_PER_THREAD / 2; ++store_pair) {
              const int32_t elem_lo = store_pair * 2;
              const int32_t out_col = acc_out_col(elem_lo, lane_id);
              const int32_t global_col = static_cast<int32_t>(blockIdx.y * TILE_SIZE_COL) + out_col;
              const uint8_t v0 = static_cast<uint8_t>(static_cast<Target>(acc_reg[M - 1][row_in_chunk][elem_lo]));
              const uint8_t v1 = static_cast<uint8_t>(static_cast<Target>(acc_reg[M - 1][row_in_chunk][elem_lo + 1]));
              *reinterpret_cast<uint16_t *>(&slice_out[global_row * ldm + global_col]) =
                  static_cast<uint16_t>(v0) | (static_cast<uint16_t>(v1) << 8);
            }
            if (flush_pos == 0) {
#pragma unroll
              for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem)
                max_abs_acc = max(max_abs_acc, abs(acc_reg[M - 1][row_in_chunk][elem]));
            }
          } else {
#pragma unroll
            for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem) {
              int32_t out_col = acc_out_col(elem, lane_id);
              int32_t global_col = static_cast<int32_t>(blockIdx.y * TILE_SIZE_COL) + out_col;

              bool in_grid = (global_row < static_cast<int32_t>(height) && global_col < static_cast<int32_t>(ldm));
              bool is_interior = in_grid && (global_row >= HALO && global_row < static_cast<int32_t>(height) - HALO &&
                                             global_col >= HALO && global_col < static_cast<int32_t>(ldm) - HALO);

              if (is_interior) {
                slice_out[global_row * ldm + global_col] = static_cast<Target>(acc_reg[M - 1][row_in_chunk][elem]);
              }

              if (flush_pos == 0 && is_interior) {
                max_abs_acc = max(max_abs_acc, abs(acc_reg[M - 1][row_in_chunk][elem]));
              }
            }
          }
        }
      } // end row_in_chunk cleanup
    } // end flush_pos loop

    // Load next inactive chunk (no save needed, cleanup is terminal)
    if (cleanup_pass < NUM_CHUNKS - 1) {
#pragma unroll
      for (int32_t ws = 0; ws < M; ++ws)
#pragma unroll
        for (int32_t r = 0; r < ROWS_PER_CHUNK; ++r)
#pragma unroll
          for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem)
            acc_reg[ws][r][elem] = saved_acc[cleanup_pass][ws][r][elem];

#pragma unroll
      for (int32_t r = 0; r < ROWS_PER_CHUNK; ++r)
#pragma unroll
        for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem)
          carry_reg[r][elem] = saved_carry[cleanup_pass][r][elem];
    }
  } // end cleanup pass loop

  // ====================================================================
  // BOUNDARY HANDLING: emulated_gain for non-interior elements.
  // Streaming formulation: processes output positions from LSB to MSB,
  // fusing accumulation with carry propagation. Uses O(1) stack
  // (boundary_carry + boundary_accum) instead of O(total_slices) array.
  // ====================================================================
  if (boundary.n_slices > 0) {
    for (int32_t tile_row = 0; tile_row < MMA_N; ++tile_row) {
      int32_t out_row_local = warp_id * MMA_N + tile_row;
      int32_t global_row = static_cast<int32_t>(blockIdx.x * TILE_SIZE_ROW) + out_row_local;

      for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem) {
        int32_t local_col = lane_id % LANES_PER_ROW;
        int32_t local_row = (lane_id / LANES_PER_ROW) + elem * GROUPS;
        int32_t out_col = local_row * LANES_PER_ROW + local_col;
        int32_t global_col = static_cast<int32_t>(blockIdx.y * TILE_SIZE_COL) + out_col;

        bool in_grid = (global_row < static_cast<int32_t>(height) && global_col < static_cast<int32_t>(ldm));
        bool is_interior = in_grid && (global_row >= HALO && global_row < static_cast<int32_t>(height) - HALO &&
                                       global_col >= HALO && global_col < static_cast<int32_t>(ldm) - HALO);

        if (in_grid && !is_interior) {
          size_t global_offset = static_cast<size_t>(global_row) * ldm + global_col;

          Acc boundary_carry = static_cast<Acc>(0);
          Acc boundary_accum = static_cast<Acc>(0);

          for (int32_t slice_pos = total_slices - 1; slice_pos >= 0; --slice_pos) {
            boundary_accum = boundary_carry;
            boundary_carry = static_cast<Acc>(0);

            for (int32_t gain_idx = 0; gain_idx < boundary.n_slices; ++gain_idx) {
              int32_t input_slice = slice_pos - gain_idx;
              if (input_slice >= 0 && !(input_slice == 0 && just_shifted)) {
                int32_t phys_in = phys_lut[input_slice];
                boundary_accum += static_cast<Acc>(in[slice_stride * phys_in + global_offset]) *
                                  static_cast<int32_t>(boundary.emulated_gain[gain_idx]);
              }
            }

            int32_t phys_out = phys_lut[slice_pos];
            Target *out_ptr = out + slice_stride * phys_out;
            if (slice_pos > 0) {
              Target digit = static_cast<Target>(boundary_accum);
              out_ptr[global_offset] = digit;
              boundary_carry = (boundary_accum - static_cast<Acc>(digit)) >> (CHAR_BIT * sizeof(Target) - 1);
            } else {
              out_ptr[global_offset] = static_cast<Target>(boundary_accum);
            }
          }

          max_abs_acc = max(max_abs_acc, abs(boundary_accum));
        }
      }
    }
  }

  // ====================================================================
  // OVERFLOW DETECTION (matches stencil_v1 pattern)
  // ====================================================================
  __syncthreads();
  int32_t has_overflow = __any_sync(__activemask(), max_abs_acc > static_cast<Acc>(threshold));
  if (lane_id == 0)
    sm_overflow[warp_id] = has_overflow;
  __syncthreads();

  if (tid < WARP_SIZE) {
    int32_t any_overflow = (tid < NUM_WARPS) ? sm_overflow[tid] : 0;
    any_overflow = __any_sync(__activemask(), any_overflow);
    if (any_overflow && tid == 0) {
      atomicExch(&desc->shift_required, 1);
    }
  }
}
} // namespace ozigpu

#endif // HEADER_OZI_STENCIL_CONV
