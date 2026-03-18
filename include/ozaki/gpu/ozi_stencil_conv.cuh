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
} // namespace ozigpu

#endif // HEADER_OZI_STENCIL_CONV
