#ifndef HEADER_OZI_STENCIL_CONV
#define HEADER_OZI_STENCIL_CONV

#include "../core/oztraits.hpp"
#include "ozi.cuh"
#include "ozi_kernels.cuh"

#include <climits>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <cuda.h>
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

constexpr int32_t COEFF_LD = MMA_N;                      // = 8 (weight matrix leading dim)
constexpr int32_t WT_SIZE = MMA_NUM * MMA_K * COEFF_LD;  // per-tessellation weight size in bytes (512 bytes)
constexpr int32_t PACKED_WT_WORDS = MMA_NUM * WARP_SIZE; // = 128 (uint32 words per tessellation per slice)
static_assert(PACKED_WT_WORDS * 4 == WT_SIZE, "Packed weight size must match WT_SIZE in bytes");
constexpr int32_t ACC_TILE_SIZE = MMA_M * MMA_N; // = 128 (elements per MMA tile)
constexpr int32_t ELEMS_PER_THREAD = ACC_TILE_SIZE / WARP_SIZE;
constexpr int32_t MAX_M = 12;           // max supported stencil slices
constexpr int32_t MAX_DATA_SLICES = 24; // matches StencilSlices::MAX_DATA_SLICES

// TMA row-major layout constants (Section 5.4 of TMA.md).
// Each tessellation stores [128 bytes/row x (DTR+1) rows] in plain row-major order.
// The +1 row absorbs MMA overreads at K positions with zero weights (Section 5.6).
constexpr int32_t SMEM_ROW_BYTES = 128;                               // = TILE_SIZE_COL
constexpr int32_t SMEM_ROWS_TMA = DATA_TILE_SIZE_ROW + 1;             // 39 for NUM_WARPS=4
constexpr int32_t SMEM_TESS_BYTES = SMEM_ROW_BYTES * SMEM_ROWS_TMA;   // 4,992 B per tess
constexpr int32_t TMA_TX_BYTES = SMEM_ROW_BYTES * DATA_TILE_SIZE_ROW; // 4,864 B per TMA call

// TMA split-buffer constants: cp.async.bulk.tensor.2d requires the dim0 (column)
// coordinate to be 16-byte aligned for SWIZZLE_NONE. Since data_col_start includes
// a HALO offset that is not 16-aligned, we round down to the nearest 16-byte boundary
// for the main 128-byte load and issue a supplementary 16-byte load at
// main_aligned + 128 to cover the tail columns past the main buffer.
// A-fragment reads use byte-level access when the halo offset is not 4-byte-aligned
// (upper tessellation, halo_off=13) to avoid misaligned uint32_t loads.
constexpr int32_t TMA_EXTRA_TILE = 16; // extra tile width (bytes)
// Pad to 128-byte multiple so sm_extra[1] stays 128-byte aligned (TMA requirement).
constexpr int32_t SMEM_EXTRA_BYTES = ((TMA_EXTRA_TILE * SMEM_ROWS_TMA + 127) / 128) * 128; // 640 B per tess
constexpr int32_t TMA_EXTRA_TX_BYTES = TMA_EXTRA_TILE * DATA_TILE_SIZE_ROW;                // 608 B per TMA call
// Compile-time halo offsets: how far data_col_start is past the 16-aligned TMA coord.
constexpr int32_t UPPER_HALO_OFF = (-HALO) & 15;               // = 13 for HALO=3
constexpr int32_t LOWER_HALO_OFF = (STENCIL_SIDE - HALO) & 15; // = 4 for HALO=3

// Chunking constants: the 8 output rows (MMA_N) per warp are split into chunks.
// Each chunk is processed serially through all data slices; accumulators live
// entirely in registers with no local-memory spills.
constexpr int32_t ROWS_PER_CHUNK = 2;
constexpr int32_t NUM_CHUNKS = MMA_N / ROWS_PER_CHUNK; // = 4
static_assert(MMA_N % ROWS_PER_CHUNK == 0, "MMA_N must be divisible by ROWS_PER_CHUNK");

// ========== cvstencil_v2 constants (chunk-free, double-buffered TMA) ==========
constexpr int32_t SLICES_PER_BATCH = 2;
constexpr int32_t NUM_BUFFER_STAGES = 2;
constexpr int32_t TESS_PER_STAGE = SLICES_PER_BATCH * 2; // 4

// Dynamic shared memory for double-buffered data tiles
constexpr int32_t DYN_SMEM_DATA = NUM_BUFFER_STAGES * TESS_PER_STAGE * SMEM_TESS_BYTES;   // 39,936 B
constexpr int32_t DYN_SMEM_EXTRA = NUM_BUFFER_STAGES * TESS_PER_STAGE * SMEM_EXTRA_BYTES; // 5,120 B
constexpr int32_t DYN_SMEM_BYTES = DYN_SMEM_DATA + DYN_SMEM_EXTRA;                        // 45,056 B

// ========== Boundary gain data (passed by value to kernel) ==========
struct BoundaryGainData {
  int8_t emulated_gain[MAX_DATA_SLICES];
  int32_t n_slices;
};

// Create a 2D TMA tensor descriptor for the stacked int8 slice buffer.
// globalDim = [ldm, total_height], boxDim = [tile_width, DTR].
// Requires ldm % 16 == 0. Uses SWIZZLE_NONE with FILL_NONE.
inline CUresult create_tma_descriptor_impl(CUtensorMap *out, const int8_t *base_ptr, size_t total_height, size_t ldm,
                                           uint32_t tile_width) {
  assert(ldm % 16 == 0 && "TMA requires ldm to be a multiple of 16");
  assert(reinterpret_cast<uintptr_t>(base_ptr) % 16 == 0 && "TMA requires 16-byte-aligned base pointer");
  assert(tile_width % 16 == 0 && tile_width <= 128 && "TMA tile width must be a multiple of 16 and <= 128");

  uint64_t gDim[2] = {ldm, total_height};
  uint64_t gStrides[1] = {ldm};
  uint32_t tileDim[2] = {tile_width, static_cast<uint32_t>(DATA_TILE_SIZE_ROW)};
  uint32_t elemStrides[2] = {1, 1};
  return cuTensorMapEncodeTiled(out, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
                                const_cast<void *>(static_cast<const void *>(base_ptr)), gDim, gStrides, tileDim,
                                elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// Ping-pong pair of main + extra TMA descriptors (host-side, passed by value via __grid_constant__).
struct TmaDescriptors {
  CUtensorMap h_main[2] = {};
  CUtensorMap h_extra[2] = {};

  void swap() {
    std::swap(h_main[0], h_main[1]);
    std::swap(h_extra[0], h_extra[1]);
  }

  void free() {} // No-op: host-side objects, no device allocation.
};

// Create main (128B) + extra (16B) TMA descriptors for two ping-pong slice buffers.
// Returns true on success. Descriptors are host-side and passed to the kernel by
// value via __grid_constant__.
inline bool create_tma_descriptors(TmaDescriptors &out, const int8_t *buf_a, const int8_t *buf_b, size_t total_height,
                                   size_t ldm) {
  if (ldm % 16 != 0 || ldm < SMEM_ROW_BYTES)
    return false;
  if (create_tma_descriptor_impl(&out.h_main[0], buf_a, total_height, ldm, SMEM_ROW_BYTES) != CUDA_SUCCESS ||
      create_tma_descriptor_impl(&out.h_main[1], buf_b, total_height, ldm, SMEM_ROW_BYTES) != CUDA_SUCCESS ||
      create_tma_descriptor_impl(&out.h_extra[0], buf_a, total_height, ldm, TMA_EXTRA_TILE) != CUDA_SUCCESS ||
      create_tma_descriptor_impl(&out.h_extra[1], buf_b, total_height, ldm, TMA_EXTRA_TILE) != CUDA_SUCCESS)
    return false;
  return true;
}

__device__ __forceinline__ void shmem_barrier_init(uint64_t *bar, uint32_t arrive_count) {
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr), "r"(arrive_count));
}

__device__ __forceinline__ void shmem_barrier_arrive_expect_tx(uint64_t *bar, uint32_t tx_count) {
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_addr), "r"(tx_count));
}

__device__ __forceinline__ void shmem_barrier_spin(uint64_t *bar, uint32_t phase_bit) {
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{\n\t"
               ".reg .pred P_%=;\n\t"
               "WAIT_%=:\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 P_%=, [%0], %1;\n\t"
               "@!P_%=  bra WAIT_%=;\n\t"
               "}\n\t" ::"r"(smem_addr),
               "r"(phase_bit)
               : "memory");
}

__device__ __forceinline__ void tma_load_tile_2d(void *smem_dst, const void *tensor_map, int32_t col, int32_t row,
                                                 uint64_t *bar) {
  uint32_t smem_dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint64_t tmap_addr = reinterpret_cast<uint64_t>(tensor_map);

  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_dst_addr),
               "l"(tmap_addr),     // 64-bit address of CUtensorMap
               "r"(col), "r"(row), // {dim0_coord, dim1_coord}
               "r"(bar_addr)
               : "memory");
}

/** Load A-fragment from split TMA buffers (main 128B + extra 16B per row).
 *
 *  The main TMA loads at a 16-byte-aligned column (floor16(data_col_start)),
 *  so the useful data starts at byte HALO_OFF within each 128-byte smem row.
 *  The extra buffer covers the tail beyond byte 128 of the main buffer.
 *
 *  When HALO_OFF is 4-byte-aligned (lower tessellation, HALO_OFF=4), direct
 *  uint32_t reads are used. When HALO_OFF is not 4-byte-aligned (upper
 *  tessellation, HALO_OFF=13), two bracketing aligned uint32_t loads are
 *  combined via __byte_perm (PTX prmt) to extract the misaligned 4 bytes.
 *  The sm_main/sm_extra boundary is resolved per-address with a conditional
 *  that compiles to a SELP (no branch divergence within the warp).
 */
template <int32_t HALO_OFF>
__device__ __forceinline__ void load_a_frag_tma_split(uint32_t (&A)[2], const int8_t *sm_main, const int8_t *sm_extra,
                                                      int32_t k_start, int32_t lane_id) {
  const int32_t groupID = lane_id >> 2;
  const int32_t tIG = lane_id & 3;
  const int32_t col_offset = (tIG & 1) * 4;
  const int32_t row_main = ((k_start >> 3) + (tIG >> 1)) * SMEM_ROW_BYTES;
  const int32_t row_extra = ((k_start >> 3) + (tIG >> 1)) * TMA_EXTRA_TILE;

#pragma unroll
  for (int32_t r = 0; r < 2; ++r) {
    const int32_t m = groupID + r * 8;
    const int32_t eff_col = HALO_OFF + m * 8 + col_offset;

    if constexpr ((HALO_OFF & 3) == 0) {
      // Aligned path: HALO_OFF is 4-byte-aligned, so eff_col is always 4-byte-aligned.
      if (eff_col < SMEM_ROW_BYTES) {
        A[r] = *reinterpret_cast<const uint32_t *>(sm_main + row_main + eff_col);
      } else {
        A[r] = *reinterpret_cast<const uint32_t *>(sm_extra + row_extra + (eff_col - SMEM_ROW_BYTES));
      }
    } else {
      // Misaligned path: eff_col % 4 == (HALO_OFF & 3), a non-zero constant.
      // Load two aligned uint32 words that bracket the desired 4 bytes,
      // then use __byte_perm to extract the correctly shifted result.
      // This replaces 4 byte-level loads with 2 aligned word loads + 1 prmt.
      constexpr int32_t MISALIGN = HALO_OFF & 3;
      constexpr uint32_t PRMT_SEL = MISALIGN | ((MISALIGN + 1) << 4) | ((MISALIGN + 2) << 8) | ((MISALIGN + 3) << 12);

      const int32_t aligned = eff_col - MISALIGN; // 4-byte-aligned floor of eff_col
      const int32_t aligned_next = aligned + 4;

      // Each aligned address resolves independently to sm_main or sm_extra.
      const uint32_t w0 = (aligned < SMEM_ROW_BYTES)
                              ? *reinterpret_cast<const uint32_t *>(sm_main + row_main + aligned)
                              : *reinterpret_cast<const uint32_t *>(sm_extra + row_extra + (aligned - SMEM_ROW_BYTES));
      const uint32_t w1 =
          (aligned_next < SMEM_ROW_BYTES)
              ? *reinterpret_cast<const uint32_t *>(sm_main + row_main + aligned_next)
              : *reinterpret_cast<const uint32_t *>(sm_extra + row_extra + (aligned_next - SMEM_ROW_BYTES));

      A[r] = __byte_perm(w0, w1, PRMT_SEL);
    }
  }
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

// ========== cvstencil_v2 helper: MMA accumulate for one tile_row ==========

/** Run 2 tessellations x MMA_NUM k-tiles of mma.sync, accumulating into `target`.
 *  A_frags must already be loaded for this tile_row. `ws` selects the weight slice. */
__device__ __forceinline__ void mma_accumulate_row(int32_t (&target)[ELEMS_PER_THREAD],
                                                   const uint32_t (&A_frags)[2][MMA_NUM][2], const uint32_t *sm_weight,
                                                   int32_t ws, int32_t lane_id) {
  uint32_t mma_accum[ELEMS_PER_THREAD];
#pragma unroll
  for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem)
    mma_accum[elem] = static_cast<uint32_t>(target[elem]);

#pragma unroll
  for (int32_t tess = 0; tess < 2; ++tess) {
#pragma unroll
    for (int32_t k_tile = 0; k_tile < MMA_NUM; ++k_tile) {
      uint32_t B = sm_weight[(ws * 2 + tess) * PACKED_WT_WORDS + k_tile * WARP_SIZE + lane_id];
      asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                   "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                   : "+r"(mma_accum[0]), "+r"(mma_accum[1]), "+r"(mma_accum[2]), "+r"(mma_accum[3])
                   : "r"(A_frags[tess][k_tile][0]), "r"(A_frags[tess][k_tile][1]), "r"(B));
    }
  }

#pragma unroll
  for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem)
    target[elem] = static_cast<int32_t>(mma_accum[elem]);
}

/** Load A-fragments from a specific slot within a double-buffered dynamic smem layout.
 *  stage: 0 or 1 (double-buffer stage)
 *  slice_in_batch: 0 or 1 (which data slice within the batch)
 *  tess: 0 (upper) or 1 (lower) tessellation */
__device__ __forceinline__ void load_a_frags_for_slot(uint32_t (&A_frags)[2][MMA_NUM][2], const int8_t *dyn_smem,
                                                      int32_t stage, int32_t slice_in_batch, int32_t k_base,
                                                      int32_t lane_id) {
#pragma unroll
  for (int32_t tess = 0; tess < 2; ++tess) {
    int32_t slot = slice_in_batch * 2 + tess; // 0..3
    const int8_t *sm_data_ptr = dyn_smem + (stage * TESS_PER_STAGE + slot) * SMEM_TESS_BYTES;
    const int8_t *sm_extra_ptr = dyn_smem + DYN_SMEM_DATA + (stage * TESS_PER_STAGE + slot) * SMEM_EXTRA_BYTES;

#pragma unroll
    for (int32_t k_tile = 0; k_tile < MMA_NUM; ++k_tile) {
      if (tess == 0)
        load_a_frag_tma_split<UPPER_HALO_OFF>(A_frags[0][k_tile], sm_data_ptr, sm_extra_ptr, k_base + k_tile * MMA_K,
                                              lane_id);
      else
        load_a_frag_tma_split<LOWER_HALO_OFF>(A_frags[1][k_tile], sm_data_ptr, sm_extra_ptr, k_base + k_tile * MMA_K,
                                              lane_id);
    }
  }
}

/** Truncation-store + carry for one tile_row of C.
 *  Stores int8(C[e]) to output, then overwrites C with carry: C[e] = (C[e] - int32(digit)) >> 7.
 *  When is_msb is true, skips carry and records max_abs_acc. */
template <typename Target, typename Acc>
__device__ __forceinline__ void trunc_store_row(int32_t (&C)[ELEMS_PER_THREAD], int32_t &max_abs_acc, Target *slice_out,
                                                size_t ldm, int32_t global_row, int32_t tile_col_base,
                                                bool tile_in_bounds, bool is_interior_row, bool is_msb, int32_t lane_id,
                                                size_t height) {
  if (tile_in_bounds) {
#pragma unroll
    for (int32_t store_pair = 0; store_pair < ELEMS_PER_THREAD / 2; ++store_pair) {
      const int32_t elem_lo = store_pair * 2;
      const int32_t out_col = acc_out_col(elem_lo, lane_id);
      const int32_t global_col = tile_col_base + out_col;
      const uint8_t v0 = static_cast<uint8_t>(static_cast<Target>(C[elem_lo]));
      const uint8_t v1 = static_cast<uint8_t>(static_cast<Target>(C[elem_lo + 1]));
      *reinterpret_cast<uint16_t *>(&slice_out[global_row * ldm + global_col]) =
          static_cast<uint16_t>(v0) | (static_cast<uint16_t>(v1) << 8);
    }
    if (is_msb) {
#pragma unroll
      for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem)
        max_abs_acc = max(max_abs_acc, abs(C[elem]));
    }
  } else {
#pragma unroll
    for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem) {
      int32_t out_col = acc_out_col(elem, lane_id);
      int32_t global_col = tile_col_base + out_col;

      bool in_grid = (global_row < static_cast<int32_t>(height) && global_col < static_cast<int32_t>(ldm));
      bool is_interior =
          in_grid && is_interior_row && (global_col >= HALO && global_col < static_cast<int32_t>(ldm) - HALO);

      if (is_interior)
        slice_out[global_row * ldm + global_col] = static_cast<Target>(C[elem]);

      if (is_msb && is_interior)
        max_abs_acc = max(max_abs_acc, abs(C[elem]));
    }
  }

  // Compute carry in-place (skip for MSB)
  if (!is_msb) {
#pragma unroll
    for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem) {
      Target trunc = static_cast<Target>(C[elem]);
      C[elem] = (C[elem] - static_cast<Acc>(trunc)) >> (CHAR_BIT * sizeof(Target) - 1);
    }
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
__global__ __launch_bounds__(convstencil::NUM_WARPS *WARP_SIZE) void cvstencil(
    typename O::target_type *__restrict__ out, const typename O::target_type *__restrict__ in,
    const typename O::target_type *__restrict__ weight, const size_t ldm, const size_t height,
    OzakiDescriptor<O> *__restrict__ desc, __grid_constant__ const convstencil::BoundaryGainData boundary,
    __grid_constant__ const CUtensorMap d_t_map, __grid_constant__ const CUtensorMap d_t_map_extra) {
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
  // TMA row-major layout: main buffer 128 bytes/row, extra buffer 16 bytes/row.
  // DTR+1 rows per tessellation (extra row absorbs MMA overreads at zero-weight K).
  __shared__ __align__(128) int8_t sm_data[2][SMEM_TESS_BYTES];
  __shared__ __align__(128) int8_t sm_extra[2][SMEM_EXTRA_BYTES];
  __shared__ __align__(8) uint64_t tma_mbar;
  uint32_t phase_bit = 0;
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

  if (tid == 0)
    shmem_barrier_init(&tma_mbar, 1); // 1 arrival = thread 0 dispatches TMA

  // Load weight matrices from global to shared memory (eliminates repeated
  // global reads during the MMA inner loop).
  {
    const uint32_t *wt_src = reinterpret_cast<const uint32_t *>(weight);
    constexpr int32_t WT_WORDS = M * 2 * PACKED_WT_WORDS;
    for (int32_t idx = tid; idx < WT_WORDS; idx += blockDim.x)
      sm_weight[idx] = wt_src[idx];
  }

  __syncthreads(); // Ensure weights loaded and mbarrier initialized before main loop

  // ====================================================================
  // CHUNK-SERIAL PROCESSING
  //
  // The 8 output rows per warp are split into NUM_CHUNKS chunks of
  // ROWS_PER_CHUNK rows. Each chunk is processed independently through
  // all data slices before moving to the next chunk. sm_data is reloaded
  // for each chunk (4x TMA) but no saved state or swaps are needed.
  // ====================================================================

  for (int32_t chunk = 0; chunk < NUM_CHUNKS; ++chunk) {

    // Fresh accumulators for this chunk
    int32_t acc_reg[M][ROWS_PER_CHUNK][ELEMS_PER_THREAD] = {0};
    int32_t carry_reg[ROWS_PER_CHUNK][ELEMS_PER_THREAD] = {0};

    // ---- Main data-slice loop ----
    for (int32_t slice_idx = total_slices - 1; slice_idx >= 0; --slice_idx) {
      const int32_t physical_slice = phys_lut[slice_idx];
      const bool skip_slice = (slice_idx == 0 && just_shifted);

      // Load input data into shared memory via TMA.
      // OOB fill (FILL_NONE) zero-fills boundary tiles, so no tile_in_bounds guard needed.
      if (!skip_slice) {
        const int32_t aligned_upper = data_col_start & ~15;
        const int32_t aligned_lower = (data_col_start + STENCIL_SIDE) & ~15;
        if (tid == 0) {
          shmem_barrier_arrive_expect_tx(&tma_mbar, 2 * TMA_TX_BYTES + 2 * TMA_EXTRA_TX_BYTES);
          const int32_t tma_row = physical_slice * static_cast<int32_t>(height) + data_row_start;
          tma_load_tile_2d(sm_data[0], &d_t_map, aligned_upper, tma_row, &tma_mbar);
          tma_load_tile_2d(sm_data[1], &d_t_map, aligned_lower, tma_row, &tma_mbar);
          tma_load_tile_2d(sm_extra[0], &d_t_map_extra, aligned_upper + SMEM_ROW_BYTES, tma_row, &tma_mbar);
          tma_load_tile_2d(sm_extra[1], &d_t_map_extra, aligned_lower + SMEM_ROW_BYTES, tma_row, &tma_mbar);
        }
        shmem_barrier_spin(&tma_mbar, phase_bit);
        phase_bit ^= 1;
      }

      // Process ONLY this chunk — no pass loop, no swaps
      for (int32_t row_in_chunk = 0; row_in_chunk < ROWS_PER_CHUNK; ++row_in_chunk) {
        const int32_t tile_row = chunk * ROWS_PER_CHUNK + row_in_chunk;

        // Advance sliding window: shift accumulators and inject carry
        advance_sliding_window<M>(acc_reg, carry_reg, row_in_chunk);

        // Tensor core matrix multiply (inline PTX m16n8k16)
        if (!skip_slice) {
          const int32_t k_base = (warp_id * MMA_N + tile_row) * STRIDE;

          // Pre-load A fragments once: they depend on (tess, k_tile, k_base, lane_id)
          // but NOT on ws, so loading inside the ws loop would be redundant.
          // 2 tessellations × MMA_NUM k-tiles × 2 registers = 16 uint32 registers.
          uint32_t A_frags[2][MMA_NUM][2];
#pragma unroll
          for (int32_t tess = 0; tess < 2; ++tess) {
#pragma unroll
            for (int32_t k_tile = 0; k_tile < MMA_NUM; ++k_tile) {
              if (tess == 0)
                load_a_frag_tma_split<UPPER_HALO_OFF>(A_frags[0][k_tile], sm_data[0], sm_extra[0],
                                                      k_base + k_tile * MMA_K, lane_id);
              else
                load_a_frag_tma_split<LOWER_HALO_OFF>(A_frags[1][k_tile], sm_data[1], sm_extra[1],
                                                      k_base + k_tile * MMA_K, lane_id);
            }
          }

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
                uint32_t B = sm_weight[(ws * 2 + tess) * PACKED_WT_WORDS + k_tile * WARP_SIZE + lane_id];

                asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                             "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                             : "+r"(mma_accum[0]), "+r"(mma_accum[1]), "+r"(mma_accum[2]), "+r"(mma_accum[3])
                             : "r"(A_frags[tess][k_tile][0]), "r"(A_frags[tess][k_tile][1]), "r"(B));
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

      __syncthreads(); // protect sm_data for next TMA load

    } // end slice loop

    // ---- Cleanup: flush M-1 remaining positions for this chunk ----
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

  } // end chunk loop

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

namespace convstencil {
/** Issue TMA loads for one batch of data slices into a double-buffer stage.
 *  All parameters passed by value to avoid address-taking (stack spills). */
__device__ __forceinline__ void issue_tma_for_batch(int32_t batch, int32_t stage, int8_t *dyn_smem, uint64_t *tma_mbar,
                                                    const CUtensorMap *d_t_map, const CUtensorMap *d_t_map_extra,
                                                    const int32_t *phys_lut, int32_t total_slices, int32_t num_batches,
                                                    int32_t just_shifted, int32_t height, int32_t data_row_start,
                                                    int32_t aligned_upper, int32_t aligned_lower) {
  int32_t slices_in_batch = (batch == num_batches - 1 && (total_slices % SLICES_PER_BATCH) != 0)
                                ? (total_slices % SLICES_PER_BATCH)
                                : SLICES_PER_BATCH;

  int32_t active_slices = 0;
  for (int32_t s = 0; s < slices_in_batch; ++s) {
    int32_t data_logical = total_slices - 1 - (batch * SLICES_PER_BATCH + s);
    if (data_logical >= 0 && !(data_logical == 0 && just_shifted))
      ++active_slices;
  }

  uint32_t tx = active_slices * (2 * TMA_TX_BYTES + 2 * TMA_EXTRA_TX_BYTES);
  shmem_barrier_arrive_expect_tx(&tma_mbar[stage], tx);

  for (int32_t s = 0; s < slices_in_batch; ++s) {
    int32_t data_logical = total_slices - 1 - (batch * SLICES_PER_BATCH + s);
    if (data_logical < 0 || (data_logical == 0 && just_shifted))
      continue;

    int32_t physical_slice = phys_lut[data_logical];
    int32_t tma_row = physical_slice * height + data_row_start;

    int32_t base_slot = s * 2;
    int8_t *data_upper = dyn_smem + (stage * TESS_PER_STAGE + base_slot + 0) * SMEM_TESS_BYTES;
    int8_t *data_lower = dyn_smem + (stage * TESS_PER_STAGE + base_slot + 1) * SMEM_TESS_BYTES;
    int8_t *extra_upper = dyn_smem + DYN_SMEM_DATA + (stage * TESS_PER_STAGE + base_slot + 0) * SMEM_EXTRA_BYTES;
    int8_t *extra_lower = dyn_smem + DYN_SMEM_DATA + (stage * TESS_PER_STAGE + base_slot + 1) * SMEM_EXTRA_BYTES;

    tma_load_tile_2d(data_upper, d_t_map, aligned_upper, tma_row, &tma_mbar[stage]);
    tma_load_tile_2d(data_lower, d_t_map, aligned_lower, tma_row, &tma_mbar[stage]);
    tma_load_tile_2d(extra_upper, d_t_map_extra, aligned_upper + SMEM_ROW_BYTES, tma_row, &tma_mbar[stage]);
    tma_load_tile_2d(extra_lower, d_t_map_extra, aligned_lower + SMEM_ROW_BYTES, tma_row, &tma_mbar[stage]);
  }
}
} // namespace convstencil

/**
 * Chunk-free ConvStencil kernel with double-buffered TMA and 1x bandwidth.
 *
 * Eliminates the 4x TMA replay of the chunked cvstencil by processing all 8
 * tile_rows per data slice without reloading. Data slices are loaded in pairs
 * ("batches") with double-buffered TMA for latency hiding.
 *
 * Uses a single accumulator C (plus C_next for M>=3) instead of M sliding-
 * window accumulators, reducing register pressure to enable 4 blocks/SM.
 * Carry propagation happens in-place: trunc_store_row extracts carry into C.
 */
template <OzakiPair O, int32_t M = 1>
  requires(std::is_same_v<typename O::target_type, int8_t>) && (M >= 1)
// clang-format off
__global__ __launch_bounds__(convstencil::NUM_WARPS *WARP_SIZE, 4) void cvstencil_v2(
    typename O::target_type *__restrict__ out,
    const typename O::target_type *__restrict__ in,
    const typename O::target_type *__restrict__ weight,
    const size_t ldm, const size_t height,
    OzakiDescriptor<O> *__restrict__ desc,
    __grid_constant__ const convstencil::BoundaryGainData boundary,
    __grid_constant__ const CUtensorMap d_t_map,
    __grid_constant__ const CUtensorMap d_t_map_extra) {
  // clang-format on
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
  extern __shared__ __align__(128) int8_t dyn_smem[];

  __shared__ __align__(16) uint32_t sm_weight[M * 2 * PACKED_WT_WORDS];
  __shared__ __align__(8) uint64_t tma_mbar[NUM_BUFFER_STAGES];
  __shared__ int32_t sm_overflow[NUM_WARPS];

  if (tid < NUM_WARPS)
    sm_overflow[tid] = 0;

  int32_t max_abs_acc = 0;

  // ======================== Tile coordinates ========================
  const int32_t data_row_start = static_cast<int32_t>(blockIdx.x * TILE_SIZE_ROW) - HALO;
  const int32_t data_col_start = static_cast<int32_t>(blockIdx.y * TILE_SIZE_COL) - HALO;

  const bool tile_in_bounds =
      (data_row_start >= 0) && (data_row_start + DATA_TILE_SIZE_ROW <= static_cast<int32_t>(height)) &&
      (data_col_start >= 0) && (data_col_start + DATA_TILE_SIZE_COL <= static_cast<int32_t>(ldm));

  const int32_t aligned_upper = data_col_start & ~15;
  const int32_t aligned_lower = (data_col_start + STENCIL_SIDE) & ~15;
  const int32_t tile_col_base = static_cast<int32_t>(blockIdx.y * TILE_SIZE_COL);

  // Initialize mbarriers
  if (tid == 0) {
    shmem_barrier_init(&tma_mbar[0], 1);
    shmem_barrier_init(&tma_mbar[1], 1);
  }

  // Load weight matrices global → shared
  {
    const uint32_t *wt_src = reinterpret_cast<const uint32_t *>(weight);
    constexpr int32_t WT_WORDS = M * 2 * PACKED_WT_WORDS;
    for (int32_t idx = tid; idx < WT_WORDS; idx += blockDim.x)
      sm_weight[idx] = wt_src[idx];
  }

  __syncthreads();

  // ====================================================================
  // SINGLE-ACCUMULATOR BATCH PROCESSING (chunk-free, double-buffered TMA)
  //
  // Data slices are loaded in pairs (batches) with double-buffered TMA.
  // A single accumulator C (plus C_next for M>=3) replaces M sliding-
  // window accumulators, reducing register pressure. Each batch produces
  // up to 2 outputs via phased MMA accumulation with in-place carry.
  // All 8 tile_rows are processed per batch (no chunking).
  //
  // Phase ordering per batch (b >= 1):
  //   M=1: [D0@W0 store] [D1@W0 store]
  //   M=2: [D0@W1 store] [D0@W0+D1@W1 store] [D1@W0 defer]
  //   M=3: [D0@W2 store] [C_next+D0@W1+D1@W2 store] [D0@W0+D1@W1 defer] [D1@W0->C_next]
  // ====================================================================

  const int32_t num_batches = (total_slices + SLICES_PER_BATCH - 1) / SLICES_PER_BATCH;

  // Single accumulator C: holds partial sums + carry (in-place via trunc_store_row)
  int32_t C[MMA_N][ELEMS_PER_THREAD] = {0};
  // Secondary accumulator for M>=3 (deferred D@W0 contribution)
  int32_t C_next[MMA_N][ELEMS_PER_THREAD] = {0};

  uint32_t phase_0 = 0, phase_1 = 0;

  // Pre-load batch 0 into stage 0
  if (tid == 0)
    issue_tma_for_batch(0, 0, dyn_smem, tma_mbar, &d_t_map, &d_t_map_extra, phys_lut, total_slices, num_batches,
                        just_shifted, static_cast<int32_t>(height), data_row_start, aligned_upper, aligned_lower);
  shmem_barrier_spin(&tma_mbar[0], phase_0);
  phase_0 ^= 1;

  // ---- Main batch loop ----
  for (int32_t batch = 0; batch < num_batches; ++batch) {
    const int32_t cur_stage = batch % NUM_BUFFER_STAGES;
    const int32_t nxt_stage = 1 - cur_stage;

    // Prefetch next batch into nxt_stage
    if (batch + 1 < num_batches) {
      if (tid == 0)
        issue_tma_for_batch(batch + 1, nxt_stage, dyn_smem, tma_mbar, &d_t_map, &d_t_map_extra, phys_lut, total_slices,
                            num_batches, just_shifted, static_cast<int32_t>(height), data_row_start, aligned_upper,
                            aligned_lower);
    }

    // How many data slices in this batch
    const int32_t slices_in_batch = (batch == num_batches - 1 && (total_slices % SLICES_PER_BATCH) != 0)
                                        ? (total_slices % SLICES_PER_BATCH)
                                        : SLICES_PER_BATCH;
    const bool has_D1 = (slices_in_batch >= 2);

    // Data slice indices (code convention: 0 = MSB, total_slices-1 = LSB)
    const int32_t ds_D0 = total_slices - 1 - batch * SLICES_PER_BATCH;
    const int32_t ds_D1 = total_slices - 2 - batch * SLICES_PER_BATCH;
    const bool skip_D0 = (ds_D0 == 0 && just_shifted);
    const bool skip_D1 = has_D1 && (ds_D1 == 0 && just_shifted);

    // Process all 8 tile_rows with phased MMA accumulation
#pragma unroll
    for (int32_t tile_row = 0; tile_row < MMA_N; ++tile_row) {
      const int32_t k_base = (warp_id * MMA_N + tile_row) * STRIDE;
      const int32_t out_row_local = warp_id * MMA_N + tile_row;
      const int32_t global_row = static_cast<int32_t>(blockIdx.x * TILE_SIZE_ROW) + out_row_local;
      const bool is_interior_row = (global_row >= HALO && global_row < static_cast<int32_t>(height) - HALO);

      // Load A-fragments for D0 and D1 (both held in registers for multi-phase use)
      uint32_t A_D0[2][MMA_NUM][2];
      if (!skip_D0)
        load_a_frags_for_slot(A_D0, dyn_smem, cur_stage, 0, k_base, lane_id);

      uint32_t A_D1[2][MMA_NUM][2];
      if (has_D1 && !skip_D1)
        load_a_frags_for_slot(A_D1, dyn_smem, cur_stage, 1, k_base, lane_id);

      // ============ Phase 1: Complete a prior output ============
      // M=1: D0@W0 -> store output[ts-1-2b]
      // M=2: D0@W1 -> store output[ts-2b]     (batch > 0 only)
      // M=3: D0@W2 -> store output[ts+1-2b]   (batch > 0 only)
      if constexpr (M == 1) {
        if (!skip_D0)
          mma_accumulate_row(C[tile_row], A_D0, sm_weight, 0, lane_id);

        const int32_t os = total_slices - 1 - 2 * batch;
        if (os >= 0 && os < total_slices) {
          const int32_t phys = phys_lut[os];
          Target *slice_out = out + slice_stride * phys;
          trunc_store_row<Target, Acc>(C[tile_row], max_abs_acc, slice_out, ldm, global_row, tile_col_base,
                                       tile_in_bounds, is_interior_row, /*is_msb=*/(os == 0), lane_id, height);
        }
      } else if constexpr (M == 2) {
        if (batch > 0) {
          if (!skip_D0)
            mma_accumulate_row(C[tile_row], A_D0, sm_weight, 1, lane_id);

          const int32_t os = total_slices - 2 * batch;
          if (os >= 0 && os < total_slices) {
            const int32_t phys = phys_lut[os];
            Target *slice_out = out + slice_stride * phys;
            trunc_store_row<Target, Acc>(C[tile_row], max_abs_acc, slice_out, ldm, global_row, tile_col_base,
                                         tile_in_bounds, is_interior_row, /*is_msb=*/(os == 0), lane_id, height);
          }
        }
      } else { // M >= 3
        if (batch > 0) {
          if (!skip_D0)
            mma_accumulate_row(C[tile_row], A_D0, sm_weight, 2, lane_id);

          const int32_t os = total_slices + 1 - 2 * batch;
          if (os >= 0 && os < total_slices) {
            const int32_t phys = phys_lut[os];
            Target *slice_out = out + slice_stride * phys;
            trunc_store_row<Target, Acc>(C[tile_row], max_abs_acc, slice_out, ldm, global_row, tile_col_base,
                                         tile_in_bounds, is_interior_row, /*is_msb=*/(os == 0), lane_id, height);
          }
        }
      }

      // ============ Phase 2: Complete current output ============
      // M=1: D1@W0 -> store output[ts-2-2b]
      // M=2: D0@W0 + D1@W1 -> store output[ts-1-2b]
      // M=3: C_next + D0@W1 + D1@W2 -> store output[ts-2b]  (batch > 0 only)
      if constexpr (M == 1) {
        if (has_D1 && !skip_D1)
          mma_accumulate_row(C[tile_row], A_D1, sm_weight, 0, lane_id);

        const int32_t os = total_slices - 2 - 2 * batch;
        if (has_D1 && os >= 0 && os < total_slices) {
          const int32_t phys = phys_lut[os];
          Target *slice_out = out + slice_stride * phys;
          trunc_store_row<Target, Acc>(C[tile_row], max_abs_acc, slice_out, ldm, global_row, tile_col_base,
                                       tile_in_bounds, is_interior_row, /*is_msb=*/(os == 0), lane_id, height);
        }
      } else if constexpr (M == 2) {
        if (!skip_D0)
          mma_accumulate_row(C[tile_row], A_D0, sm_weight, 0, lane_id);
        if (has_D1 && !skip_D1)
          mma_accumulate_row(C[tile_row], A_D1, sm_weight, 1, lane_id);

        const int32_t os = total_slices - 1 - 2 * batch;
        if (os >= 0 && os < total_slices) {
          const int32_t phys = phys_lut[os];
          Target *slice_out = out + slice_stride * phys;
          trunc_store_row<Target, Acc>(C[tile_row], max_abs_acc, slice_out, ldm, global_row, tile_col_base,
                                       tile_in_bounds, is_interior_row, /*is_msb=*/(os == 0), lane_id, height);
        }
      } else { // M >= 3
        if (batch > 0) {
          // Inject deferred C_next into C
#pragma unroll
          for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e) {
            C[tile_row][e] += C_next[tile_row][e];
            C_next[tile_row][e] = 0;
          }
          if (!skip_D0)
            mma_accumulate_row(C[tile_row], A_D0, sm_weight, 1, lane_id);
          if (has_D1 && !skip_D1)
            mma_accumulate_row(C[tile_row], A_D1, sm_weight, 2, lane_id);

          const int32_t os = total_slices - 2 * batch;
          if (os >= 0 && os < total_slices) {
            const int32_t phys = phys_lut[os];
            Target *slice_out = out + slice_stride * phys;
            trunc_store_row<Target, Acc>(C[tile_row], max_abs_acc, slice_out, ldm, global_row, tile_col_base,
                                         tile_in_bounds, is_interior_row, /*is_msb=*/(os == 0), lane_id, height);
          }
        }
      }

      // ============ Phase 3: Defer into C ============
      // M=2: C += D1@W0
      // M=3: C += D0@W0 + D1@W1
      if constexpr (M == 2) {
        if (has_D1 && !skip_D1)
          mma_accumulate_row(C[tile_row], A_D1, sm_weight, 0, lane_id);
      } else if constexpr (M >= 3) {
        if (!skip_D0)
          mma_accumulate_row(C[tile_row], A_D0, sm_weight, 0, lane_id);
        if (has_D1 && !skip_D1)
          mma_accumulate_row(C[tile_row], A_D1, sm_weight, 1, lane_id);
      }

      // ============ Phase 4: Defer into C_next (M >= 3 only) ============
      if constexpr (M >= 3) {
#pragma unroll
        for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e)
          C_next[tile_row][e] = 0;
        if (has_D1 && !skip_D1)
          mma_accumulate_row(C_next[tile_row], A_D1, sm_weight, 0, lane_id);
      }
    } // end tile_row

    __syncthreads(); // protect current stage for next batch's TMA

    // Wait for prefetched next batch
    if (batch + 1 < num_batches) {
      if (nxt_stage == 0) {
        shmem_barrier_spin(&tma_mbar[0], phase_0);
        phase_0 ^= 1;
      } else {
        shmem_barrier_spin(&tma_mbar[1], phase_1);
        phase_1 ^= 1;
      }
    }
  } // end batch loop

  // ---- Flush: M-1 remaining outputs ----
  // The batch loop stores outputs down to a certain position; the flush
  // covers the remaining M-1 most-significant positions (output[M-2]..output[0]).
  // For odd total_slices with M=3, some flush positions were already stored
  // by the last batch, so we guard with lowest_batch_os.
  if constexpr (M >= 2) {
    // Lowest output_slice stored by the batch loop's Phase 2
    const int32_t lowest_batch_os =
        (M == 2) ? (total_slices - 1 - 2 * (num_batches - 1)) : (total_slices - 2 * (num_batches - 1));

    for (int32_t flush_pos = M - 2; flush_pos >= 0; --flush_pos) {
      if (flush_pos >= lowest_batch_os || flush_pos >= total_slices)
        continue;

#pragma unroll
      for (int32_t tile_row = 0; tile_row < MMA_N; ++tile_row) {
        // For the MSB position (flush_pos == 0), inject C_next into C
        if constexpr (M >= 3) {
          if (flush_pos == 0) {
#pragma unroll
            for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e)
              C[tile_row][e] += C_next[tile_row][e];
          }
        }

        const int32_t out_row_local = warp_id * MMA_N + tile_row;
        const int32_t global_row = static_cast<int32_t>(blockIdx.x * TILE_SIZE_ROW) + out_row_local;
        const bool is_interior_row = (global_row >= HALO && global_row < static_cast<int32_t>(height) - HALO);

        const int32_t phys = phys_lut[flush_pos];
        Target *slice_out = out + slice_stride * phys;
        trunc_store_row<Target, Acc>(C[tile_row], max_abs_acc, slice_out, ldm, global_row, tile_col_base,
                                     tile_in_bounds, is_interior_row, /*is_msb=*/(flush_pos == 0), lane_id, height);
      } // end tile_row flush
    } // end flush_pos
  }

  // ====================================================================
  // BOUNDARY HANDLING (verbatim from cvstencil)
  // ====================================================================
  if (boundary.n_slices > 0) {
    for (int32_t tile_row = 0; tile_row < MMA_N; ++tile_row) {
      int32_t out_row_local = warp_id * MMA_N + tile_row;
      int32_t global_row = static_cast<int32_t>(blockIdx.x * TILE_SIZE_ROW) + out_row_local;

      for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem) {
        int32_t local_col = lane_id % MMA_N;
        int32_t local_row = (lane_id / MMA_N) + elem * (WARP_SIZE / MMA_N);
        int32_t out_col = local_row * MMA_N + local_col;
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
