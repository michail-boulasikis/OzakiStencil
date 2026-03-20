#ifndef HEADER_OZI_STENCIL_CONV_V2
#define HEADER_OZI_STENCIL_CONV_V2

#include "ozi_stencil_conv_v1.cuh"

namespace ozigpu {
namespace convstencil {

// ========== cvstencil_v2 constants (chunk-free, triple-buffered TMA) ==========
constexpr int32_t SLICES_PER_BATCH = 2;
constexpr int32_t NUM_BUFFER_STAGES = 3;
constexpr int32_t SLOTS_PER_STAGE = SLICES_PER_BATCH; // 2 (one unified buffer per slice, not per tessellation)

// Lower tessellation HALO_OFF in unified buffer: original offset + inter-tessellation gap
constexpr int32_t LOWER_HALO_OFF_UNIFIED = STENCIL_SIDE + UPPER_HALO_OFF; // 20

// 192B unified TMA tile: eliminates bank conflicts.
// With 128B rows, (128/4)%32 = 0 -> systematic 2-way bank conflicts.
// With 192B rows, (192/4)%32 = 16 -> consecutive rows offset by 16 banks, perfectly complementary.
constexpr int32_t SMEM_ROW_BYTES_V2 = 192;
constexpr int32_t SMEM_TESS_BYTES_V2 = SMEM_ROW_BYTES_V2 * SMEM_ROWS_TMA;   // 7,488 B
constexpr int32_t TMA_TX_BYTES_V2 = SMEM_ROW_BYTES_V2 * DATA_TILE_SIZE_ROW; // 7,296 B
// TMA SWIZZLE_NONE requires destination aligned to next_power_of_2(boxDim[0]) = 256 bytes.
// Pad each slot to 256-byte boundary so all TMA destinations are correctly aligned.
constexpr int32_t SMEM_SLOT_STRIDE_V2 = ((SMEM_TESS_BYTES_V2 + 255) / 256) * 256;             // 7,680 B
constexpr int32_t DYN_SMEM_BYTES = NUM_BUFFER_STAGES * SLOTS_PER_STAGE * SMEM_SLOT_STRIDE_V2; // 46,080 B

static_assert(MMA_NUM % 2 == 0, "MMA_NUM must be even for k32 pairing");

// ========== cvstencil_v2 helpers ==========

/** Load K=32 A-fragment from a unified 192B-per-row buffer.
 *  Computes row base once and reuses for both k16 halves (row_hi = row_lo + 2*ROW_STRIDE).
 *  Takes k_row (= k_start / STRIDE) directly, avoiding the multiply-then-shift roundtrip. */
template <int32_t HALO_OFF, int32_t ROW_STRIDE>
__device__ __forceinline__ void load_a_frag_k32(uint32_t (&A)[4], const int8_t *sm, int32_t k_row, int32_t lane_id) {
  const int32_t groupID = lane_id >> 2;
  const int32_t tIG = lane_id & 3;
  const int32_t col_offset = (tIG & 1) * 4;
  const int32_t row_lo = (k_row + (tIG >> 1)) * ROW_STRIDE;

#pragma unroll
  for (int32_t half = 0; half < 2; ++half) {
    const int32_t row = row_lo + half * (2 * ROW_STRIDE);
#pragma unroll
    for (int32_t r = 0; r < 2; ++r) {
      const int32_t m = groupID + r * 8;
      const int32_t eff_col = HALO_OFF + m * 8 + col_offset;

      if constexpr ((HALO_OFF & 3) == 0) {
        A[half * 2 + r] = *reinterpret_cast<const uint32_t *>(sm + row + eff_col);
      } else {
        constexpr int32_t MISALIGN = HALO_OFF & 3;
        constexpr uint32_t PRMT_SEL = MISALIGN | ((MISALIGN + 1) << 4) | ((MISALIGN + 2) << 8) | ((MISALIGN + 3) << 12);
        const int32_t aligned = eff_col - MISALIGN;
        const uint32_t w0 = *reinterpret_cast<const uint32_t *>(sm + row + aligned);
        const uint32_t w1 = *reinterpret_cast<const uint32_t *>(sm + row + aligned + 4);
        A[half * 2 + r] = __byte_perm(w0, w1, PRMT_SEL);
      }
    }
  }
}

/** Weight-reuse MMA (K=32): load B-pair once, execute m16n8k32 across all 8 tile_rows.
 *  Half-batched: processes 4 tile_rows per phase to keep A-register pressure at 16 (same as K=16). */
template <int32_t HALO_OFF>
__device__ __forceinline__ void mma_weight_reuse(int32_t (&C_all)[MMA_N][ELEMS_PER_THREAD], const int8_t *sm_data,
                                                 uint32_t B_lo, uint32_t B_hi, int32_t k32_tile, int32_t warp_id,
                                                 int32_t lane_id) {
  constexpr int32_t HALF = MMA_N / 2;
  constexpr int32_t K32_K_ROWS = 2 * MMA_K / STRIDE; // = 4
  uint32_t A_half[HALF][4];

  // ---- First half: tile_rows 0..3 ----
#pragma unroll
  for (int32_t tr = 0; tr < HALF; ++tr) {
    const int32_t k_row = warp_id * MMA_N + tr + k32_tile * K32_K_ROWS;
    load_a_frag_k32<HALO_OFF, SMEM_ROW_BYTES_V2>(A_half[tr], sm_data, k_row, lane_id);
  }
#pragma unroll
  for (int32_t tr = 0; tr < HALF; ++tr) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(C_all[tr][0]), "+r"(C_all[tr][1]), "+r"(C_all[tr][2]), "+r"(C_all[tr][3])
                 : "r"(A_half[tr][0]), "r"(A_half[tr][1]), "r"(A_half[tr][2]), "r"(A_half[tr][3]), "r"(B_lo),
                   "r"(B_hi));
  }

  // ---- Second half: tile_rows 4..7 ----
#pragma unroll
  for (int32_t tr = 0; tr < HALF; ++tr) {
    const int32_t k_row = warp_id * MMA_N + (tr + HALF) + k32_tile * K32_K_ROWS;
    load_a_frag_k32<HALO_OFF, SMEM_ROW_BYTES_V2>(A_half[tr], sm_data, k_row, lane_id);
  }
#pragma unroll
  for (int32_t tr = 0; tr < HALF; ++tr) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(C_all[tr + HALF][0]), "+r"(C_all[tr + HALF][1]), "+r"(C_all[tr + HALF][2]), "+r"(C_all[tr + HALF][3])
        : "r"(A_half[tr][0]), "r"(A_half[tr][1]), "r"(A_half[tr][2]), "r"(A_half[tr][3]), "r"(B_lo), "r"(B_hi));
  }
}

/** Store truncated int8 values from accumulator to output for one tile_row.
 *  Fast path (tile_in_bounds): vectorized uint16_t stores for element pairs.
 *  Slow path: per-element bounds checking, stores only interior cells. */
template <typename Target>
__device__ __forceinline__ void store_row(const int32_t (&C)[ELEMS_PER_THREAD], Target *slice_out, size_t ldm,
                                          int32_t global_row, int32_t tile_col_base, bool tile_in_bounds,
                                          bool is_interior_row, int32_t lane_id, size_t height) {
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
    }
  }
}

/** In-place carry extraction: C[e] = (C[e] - int8(C[e])) >> 7 */
template <typename Target, typename Acc> __device__ __forceinline__ void extract_carry(int32_t (&C)[ELEMS_PER_THREAD]) {
#pragma unroll
  for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem) {
    Target trunc = static_cast<Target>(C[elem]);
    C[elem] = (C[elem] - static_cast<Acc>(trunc)) >> (CHAR_BIT * sizeof(Target) - 1);
  }
}

/** Record max|C| for MSB overflow detection.
 *  Fast path (tile_in_bounds): all elements are interior, check unconditionally.
 *  Slow path: only check interior cells. */
__device__ __forceinline__ void check_overflow(const int32_t (&C)[ELEMS_PER_THREAD], int32_t &max_abs_acc,
                                               bool tile_in_bounds, int32_t global_row, int32_t tile_col_base,
                                               bool is_interior_row, int32_t lane_id, size_t ldm, size_t height) {
  if (tile_in_bounds) {
#pragma unroll
    for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem)
      max_abs_acc = max(max_abs_acc, abs(C[elem]));
  } else {
#pragma unroll
    for (int32_t elem = 0; elem < ELEMS_PER_THREAD; ++elem) {
      int32_t out_col = acc_out_col(elem, lane_id);
      int32_t global_col = tile_col_base + out_col;
      bool in_grid = (global_row < static_cast<int32_t>(height) && global_col < static_cast<int32_t>(ldm));
      bool is_interior =
          in_grid && is_interior_row && (global_col >= HALO && global_col < static_cast<int32_t>(ldm) - HALO);
      if (is_interior)
        max_abs_acc = max(max_abs_acc, abs(C[elem]));
    }
  }
}

/** Issue TMA loads for one batch of data slices into a triple-buffer stage.
 *  Unified 192B buffer: 1 TMA per data slice (no split main+tail).
 *  All parameters passed by value to avoid address-taking (stack spills). */
__device__ __forceinline__ void issue_tma_for_batch(int32_t batch, int32_t stage, int8_t *dyn_smem, uint64_t *tma_mbar,
                                                    const CUtensorMap *d_t_map, const int32_t *phys_lut,
                                                    int32_t total_slices, int32_t num_batches, int32_t just_shifted,
                                                    int32_t height, int32_t data_row_start, int32_t aligned_upper) {
  int32_t slices_in_batch = (batch == num_batches - 1 && (total_slices % SLICES_PER_BATCH) != 0)
                                ? (total_slices % SLICES_PER_BATCH)
                                : SLICES_PER_BATCH;

  int32_t active_slices = 0;
  for (int32_t s = 0; s < slices_in_batch; ++s) {
    int32_t data_logical = total_slices - 1 - (batch * SLICES_PER_BATCH + s);
    if (data_logical >= 0 && !(data_logical == 0 && just_shifted))
      ++active_slices;
  }

  uint32_t tx = active_slices * TMA_TX_BYTES_V2;
  shmem_barrier_arrive_expect_tx(&tma_mbar[stage], tx);

  for (int32_t s = 0; s < slices_in_batch; ++s) {
    int32_t data_logical = total_slices - 1 - (batch * SLICES_PER_BATCH + s);
    if (data_logical < 0 || (data_logical == 0 && just_shifted))
      continue;

    int32_t physical_slice = phys_lut[data_logical];
    int32_t tma_row = physical_slice * height + data_row_start;

    int8_t *data_dst = dyn_smem + (stage * SLOTS_PER_STAGE + s) * SMEM_SLOT_STRIDE_V2;
    tma_load_tile_2d(data_dst, d_t_map, aligned_upper, tma_row, &tma_mbar[stage]);
  }
}

// ========== cvstencil_v2 compile-time schedule ==========

struct MmaOp {
  int8_t data_slot;   // 0 = D0, 1 = D1
  int8_t weight_slot; // index into sm_weight
};

struct BatchStep {
  MmaOp ops[2];               // at most 2 MMA operations per step
  int8_t n_ops;               // 0, 1, or 2
  int8_t inject_cnext;        // add C_next into C and zero C_next before MMAs
  int8_t store_output;        // store C to output after MMAs
  int8_t output_offset;       // os = total_slices + output_offset - 2*batch
  int8_t accumulate_to_cnext; // accumulate into C_next (zero first), not C
  int8_t skip_first_batch;    // skip this step when batch == 0
  int8_t requires_d1;         // store additionally guarded by has_D1
};

static constexpr int32_t MAX_BATCH_STEPS = 4;

struct OutputSchedule {
  BatchStep steps[MAX_BATCH_STEPS];
  int32_t n_steps;
};

/** Build the compile-time batch schedule for a given M (number of stencil slices).
 *  Each M variant encodes the v2 phase structure as a sequence of steps. */
template <int32_t M> static constexpr OutputSchedule make_schedule() {
  OutputSchedule s{};
  if constexpr (M == 1) {
    s.n_steps = 2;
    // Step 0: D0@W0 -> store at os = ts-1-2b
    s.steps[0].ops[0] = {0, 0};
    s.steps[0].n_ops = 1;
    s.steps[0].store_output = 1;
    s.steps[0].output_offset = -1;
    // Step 1: D1@W0 -> store at os = ts-2-2b, requires_d1
    s.steps[1].ops[0] = {1, 0};
    s.steps[1].n_ops = 1;
    s.steps[1].store_output = 1;
    s.steps[1].output_offset = -2;
    s.steps[1].requires_d1 = 1;
  } else if constexpr (M == 2) {
    s.n_steps = 3;
    // Step 0: D0@W1 -> store at os = ts-2b, skip_first_batch
    s.steps[0].ops[0] = {0, 1};
    s.steps[0].n_ops = 1;
    s.steps[0].store_output = 1;
    s.steps[0].output_offset = 0;
    s.steps[0].skip_first_batch = 1;
    // Step 1: D0@W0 + D1@W1 -> store at os = ts-1-2b
    s.steps[1].ops[0] = {0, 0};
    s.steps[1].ops[1] = {1, 1};
    s.steps[1].n_ops = 2;
    s.steps[1].store_output = 1;
    s.steps[1].output_offset = -1;
    // Step 2: D1@W0 -> defer into C
    s.steps[2].ops[0] = {1, 0};
    s.steps[2].n_ops = 1;
  } else {
    s.n_steps = 4;
    // Step 0: D0@W2 -> store at os = ts+1-2b, skip_first_batch
    s.steps[0].ops[0] = {0, 2};
    s.steps[0].n_ops = 1;
    s.steps[0].store_output = 1;
    s.steps[0].output_offset = 1;
    s.steps[0].skip_first_batch = 1;
    // Step 1: inject C_next, D0@W1 + D1@W2 -> store at os = ts-2b, skip_first_batch
    s.steps[1].ops[0] = {0, 1};
    s.steps[1].ops[1] = {1, 2};
    s.steps[1].n_ops = 2;
    s.steps[1].inject_cnext = 1;
    s.steps[1].store_output = 1;
    s.steps[1].output_offset = 0;
    s.steps[1].skip_first_batch = 1;
    // Step 2: D0@W0 + D1@W1 -> defer into C
    s.steps[2].ops[0] = {0, 0};
    s.steps[2].ops[1] = {1, 1};
    s.steps[2].n_ops = 2;
    // Step 3: D1@W0 -> accumulate_to_cnext
    s.steps[3].ops[0] = {1, 0};
    s.steps[3].n_ops = 1;
    s.steps[3].accumulate_to_cnext = 1;
  }
  return s;
}

// Ping-pong pair of unified 192B TMA descriptors (host-side, passed by value via __grid_constant__).
struct TmaDescriptorsV2 {
  CUtensorMap h_main[2] = {};
  void swap() { std::swap(h_main[0], h_main[1]); }
  void free() {} // No-op: host-side objects, no device allocation.
};

// Create unified 192B TMA descriptors for two ping-pong slice buffers.
// ldm minimum stays at SMEM_ROW_BYTES (128), NOT raised to 192.
// TMA zero-fills OOB columns for integer types, so tile_width > ldm is safe.
inline bool create_tma_descriptors_v2(TmaDescriptorsV2 &out, const int8_t *buf_a, const int8_t *buf_b,
                                      size_t total_height, size_t ldm) {
  if (ldm % 16 != 0 || ldm < SMEM_ROW_BYTES)
    return false;
  if (create_tma_descriptor_impl(&out.h_main[0], buf_a, total_height, ldm, SMEM_ROW_BYTES_V2) != CUDA_SUCCESS ||
      create_tma_descriptor_impl(&out.h_main[1], buf_b, total_height, ldm, SMEM_ROW_BYTES_V2) != CUDA_SUCCESS)
    return false;
  return true;
}

} // namespace convstencil

using namespace ozi;

/**
 * Chunk-free ConvStencil kernel with triple-buffered TMA and 1x bandwidth.
 *
 * Eliminates the 4x TMA replay of the chunked cvstencil by processing all 8
 * tile_rows per data slice without reloading. Data slices are loaded in pairs
 * ("batches") with triple-buffered TMA for latency hiding.
 *
 * Uses a single accumulator C (plus C_next for M>=3) instead of M sliding-
 * window accumulators, reducing register pressure to enable 4 blocks/SM.
 * Carry propagation happens in-place: store_row + extract_carry extract carry into C.
 */
template <OzakiPair O, int32_t M = 1>
  requires(std::is_same_v<typename O::target_type, int8_t>) && (M >= 1)
// clang-format off
__global__ __launch_bounds__(convstencil::NUM_WARPS *WARP_SIZE, 1) void cvstencil_v2(
    typename O::target_type *__restrict__ out,
    const typename O::target_type *__restrict__ in,
    const typename O::target_type *__restrict__ weight,
    const size_t ldm, const size_t height,
    OzakiDescriptor<O> *__restrict__ desc,
    __grid_constant__ const convstencil::BoundaryGainData boundary,
    __grid_constant__ const CUtensorMap d_t_map) {
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
  extern __shared__ __align__(256) int8_t dyn_smem[];

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
  const int32_t tile_col_base = static_cast<int32_t>(blockIdx.y * TILE_SIZE_COL);

  // Initialize mbarriers
  if (tid == 0) {
    shmem_barrier_init(&tma_mbar[0], 1);
    shmem_barrier_init(&tma_mbar[1], 1);
    shmem_barrier_init(&tma_mbar[2], 1);
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
  // SINGLE-ACCUMULATOR BATCH PROCESSING (chunk-free, triple-buffered TMA)
  //
  // Data slices are loaded in pairs (batches) with triple-buffered TMA.
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

  uint32_t phase_0 = 0, phase_1 = 0, phase_2 = 0;

  // Pre-load batch 0 into stage 0 (blocking wait — needed immediately)
  if (tid == 0)
    issue_tma_for_batch(0, 0, dyn_smem, tma_mbar, &d_t_map, phys_lut, total_slices, num_batches, just_shifted,
                        static_cast<int32_t>(height), data_row_start, aligned_upper);
  shmem_barrier_spin(&tma_mbar[0], phase_0);
  phase_0 ^= 1;

  // Pre-load batch 1 into stage 1 (async — waited on at end of batch 0)
  if (num_batches > 1 && tid == 0)
    issue_tma_for_batch(1, 1, dyn_smem, tma_mbar, &d_t_map, phys_lut, total_slices, num_batches, just_shifted,
                        static_cast<int32_t>(height), data_row_start, aligned_upper);

  // ---- Main batch loop ----
  for (int32_t batch = 0; batch < num_batches; ++batch) {
    const int32_t cur_stage = batch % 3;
    const int32_t nxt_stage = (batch + 1) % 3;
    const int32_t far_stage = (batch + 2) % 3;

    // Prefetch batch+2 into far_stage (2-batch lookahead for 3-stage pipeline)
    if (batch + 2 < num_batches) {
      if (tid == 0)
        issue_tma_for_batch(batch + 2, far_stage, dyn_smem, tma_mbar, &d_t_map, phys_lut, total_slices, num_batches,
                            just_shifted, static_cast<int32_t>(height), data_row_start, aligned_upper);
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

    // Compute smem data pointers once per batch
    const int8_t *sm_D0 = dyn_smem + cur_stage * SLOTS_PER_STAGE * SMEM_SLOT_STRIDE_V2;
    const int8_t *sm_D1 = dyn_smem + (cur_stage * SLOTS_PER_STAGE + 1) * SMEM_SLOT_STRIDE_V2;

    // Weight-major schedule execution: weight outer, tile_row inner.
    // Each weight B-operand is loaded once and reused across all 8 tile_rows,
    // reducing weight LDS from 256 to 32 per batch (8x reduction).
    auto exec_step = [&](auto step_ic) {
      constexpr int32_t step = decltype(step_ic)::value;
      constexpr auto SCHED = make_schedule<M>();
      if constexpr (step >= SCHED.n_steps)
        return;
      constexpr auto s = SCHED.steps[step];

      if constexpr (s.skip_first_batch) {
        if (batch == 0)
          return;
      }

      // ---- Pre-step: inject_cnext / zero C_next across ALL tile_rows ----
      if constexpr (M >= 3 && s.inject_cnext) {
#pragma unroll
        for (int32_t tr = 0; tr < MMA_N; ++tr)
#pragma unroll
          for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e) {
            C[tr][e] += C_next[tr][e];
            C_next[tr][e] = 0;
          }
      }
      if constexpr (M >= 3 && s.accumulate_to_cnext) {
#pragma unroll
        for (int32_t tr = 0; tr < MMA_N; ++tr)
#pragma unroll
          for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e)
            C_next[tr][e] = 0;
      }

      // ---- MMA ops with weight reuse (inlined to keep constexpr visible to NVCC) ----
#define CVSTENCIL_V2_WEIGHT_MAJOR_OP(OP_IDX)                                                                           \
  if constexpr (s.n_ops >= (OP_IDX) + 1) {                                                                             \
    constexpr auto mop = s.ops[(OP_IDX)];                                                                              \
    const int8_t *sm_data;                                                                                             \
    bool skip;                                                                                                         \
    if constexpr (mop.data_slot == 0) {                                                                                \
      sm_data = sm_D0;                                                                                                 \
      skip = skip_D0;                                                                                                  \
    } else {                                                                                                           \
      sm_data = sm_D1;                                                                                                 \
      skip = !has_D1 || skip_D1;                                                                                       \
    }                                                                                                                  \
    if (!skip) {                                                                                                       \
      _Pragma("unroll") for (int32_t tess = 0; tess < 2; ++tess) {                                                     \
        _Pragma("unroll") for (int32_t k32 = 0; k32 < MMA_NUM / 2; ++k32) {                                            \
          uint32_t B_lo = sm_weight[(mop.weight_slot * 2 + tess) * PACKED_WT_WORDS + (k32 * 2) * WARP_SIZE + lane_id]; \
          uint32_t B_hi =                                                                                              \
              sm_weight[(mop.weight_slot * 2 + tess) * PACKED_WT_WORDS + (k32 * 2 + 1) * WARP_SIZE + lane_id];         \
          if (tess == 0) {                                                                                             \
            if constexpr (s.accumulate_to_cnext)                                                                       \
              mma_weight_reuse<UPPER_HALO_OFF>(C_next, sm_data, B_lo, B_hi, k32, warp_id, lane_id);                    \
            else                                                                                                       \
              mma_weight_reuse<UPPER_HALO_OFF>(C, sm_data, B_lo, B_hi, k32, warp_id, lane_id);                         \
          } else {                                                                                                     \
            if constexpr (s.accumulate_to_cnext)                                                                       \
              mma_weight_reuse<LOWER_HALO_OFF_UNIFIED>(C_next, sm_data, B_lo, B_hi, k32, warp_id, lane_id);            \
            else                                                                                                       \
              mma_weight_reuse<LOWER_HALO_OFF_UNIFIED>(C, sm_data, B_lo, B_hi, k32, warp_id, lane_id);                 \
          }                                                                                                            \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }
      CVSTENCIL_V2_WEIGHT_MAJOR_OP(0)
      CVSTENCIL_V2_WEIGHT_MAJOR_OP(1)
#undef CVSTENCIL_V2_WEIGHT_MAJOR_OP

      // ---- Post-step: store + carry across ALL tile_rows ----
      if constexpr (s.store_output) {
        const int32_t os = total_slices + static_cast<int32_t>(s.output_offset) - 2 * batch;
        if (os >= 0 && os < total_slices && (!s.requires_d1 || has_D1)) {
          const int32_t phys = phys_lut[os];
          Target *slice_out = out + slice_stride * phys;
#pragma unroll
          for (int32_t tr = 0; tr < MMA_N; ++tr) {
            const int32_t global_row = static_cast<int32_t>(blockIdx.x * TILE_SIZE_ROW) + warp_id * MMA_N + tr;
            const bool is_interior_row = (global_row >= HALO && global_row < static_cast<int32_t>(height) - HALO);
            store_row(C[tr], slice_out, ldm, global_row, tile_col_base, tile_in_bounds, is_interior_row, lane_id,
                      height);
            if (os == 0)
              check_overflow(C[tr], max_abs_acc, tile_in_bounds, global_row, tile_col_base, is_interior_row, lane_id,
                             ldm, height);
            else
              extract_carry<Target, Acc>(C[tr]);
          }
        }
      }
    };
    exec_step(std::integral_constant<int32_t, 0>{});
    exec_step(std::integral_constant<int32_t, 1>{});
    exec_step(std::integral_constant<int32_t, 2>{});
    exec_step(std::integral_constant<int32_t, 3>{});

    __syncthreads(); // protect cur_stage from being overwritten by far_stage's TMA

    // Wait for next batch's data (as late as possible to maximize overlap)
    if (batch + 1 < num_batches) {
      if (nxt_stage == 0) {
        shmem_barrier_spin(&tma_mbar[0], phase_0);
        phase_0 ^= 1;
      } else if (nxt_stage == 1) {
        shmem_barrier_spin(&tma_mbar[1], phase_1);
        phase_1 ^= 1;
      } else {
        shmem_barrier_spin(&tma_mbar[2], phase_2);
        phase_2 ^= 1;
      }
    }
  } // end batch loop

  // ---- Flush: M-1 remaining outputs ----
  // The batch loop stores outputs down to a certain position; the flush
  // covers the remaining M-1 most-significant positions (output[M-2]..output[0]).
  // For odd total_slices with M=3, some flush positions were already stored
  // by the last batch, so we guard with lowest_batch_os.
  if constexpr (M >= 2) {
    const int32_t lowest_batch_os =
        (M == 2) ? (total_slices - 1 - 2 * (num_batches - 1)) : (total_slices - 2 * (num_batches - 1));

    for (int32_t flush_pos = M - 2; flush_pos >= 0; --flush_pos) {
      if (flush_pos >= lowest_batch_os || flush_pos >= total_slices)
        continue;

#pragma unroll
      for (int32_t tile_row = 0; tile_row < MMA_N; ++tile_row) {
        if constexpr (M >= 3) {
          if (flush_pos == 0) {
#pragma unroll
            for (int32_t e = 0; e < ELEMS_PER_THREAD; ++e)
              C[tile_row][e] += C_next[tile_row][e];
          }
        }

        const int32_t global_row = static_cast<int32_t>(blockIdx.x * TILE_SIZE_ROW) + warp_id * MMA_N + tile_row;
        const bool is_interior_row = (global_row >= HALO && global_row < static_cast<int32_t>(height) - HALO);

        const int32_t phys = phys_lut[flush_pos];
        Target *slice_out = out + slice_stride * phys;
        store_row(C[tile_row], slice_out, ldm, global_row, tile_col_base, tile_in_bounds, is_interior_row, lane_id,
                  height);
        if (flush_pos == 0)
          check_overflow(C[tile_row], max_abs_acc, tile_in_bounds, global_row, tile_col_base, is_interior_row, lane_id,
                         ldm, height);
        else
          extract_carry<Target, Acc>(C[tile_row]);
      }
    }
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

#endif // HEADER_OZI_STENCIL_CONV_V2
