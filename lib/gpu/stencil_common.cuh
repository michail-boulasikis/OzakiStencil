// Building blocks shared by the 2D and 3D stencil kernels.
//
// Notation. The input holds `dataSlices` int8 digit planes x_0..x_{n-1} and
// the stencil holds M int8 coefficient slices w_0..w_{M-1}. Output digit d
// (a "diagonal") is the integer tensor-core product sum
//     D_d = sum_j  stencil(x_{d-j}, w_j)
// plus the carry out of diagonal d-1. Diagonals are produced from the least
// to the most significant one; each emitted diagonal keeps its low 8-bit
// digit and passes the remaining value on as carry.
#ifndef OZAKI_GPU_STENCIL_COMMON_CUH
#define OZAKI_GPU_STENCIL_COMMON_CUH

#include "input_planes.cuh"
#include <ozaki/gpu/sliced_tensor.cuh>

#include <cute/tensor.hpp>

#include <cute/algorithm/tensor_algorithms.hpp>
#include <cutlass/arch/barrier.h>

#include <cstdint>
#include <limits>

namespace ozaki::gpu::kernels {

using Mma = decltype(cute::make_tiled_mma(cute::SM80_16x8x32_S32S8S8S32_TN{}));

// Expands `function(Int<0>{}) ... function(Int<N-1>{})` without CuTe's
// intermediate lambda, so the always-inline annotations of the caller hold.
template <class Function, int... Indices>
CUDA_HOST CUDA_DEVICE CUDA_INLINE void static_for(cute::integer_sequence<int, Indices...>, Function &&function) {
  (function(cute::Int<Indices>{}), ...);
}
template <int N, class Function> CUDA_HOST CUDA_DEVICE CUDA_INLINE void static_for(Function &&function) {
  static_for(cute::make_integer_sequence<int, N>{}, function);
}

// ---------------------------------------------------------------------------
// Per-step schedule: which input planes are non-zero and which diagonals are
// stored, derived from the input descriptor and the window chosen by the
// preamble kernel.
// ---------------------------------------------------------------------------
template <int32_t M> struct Schedule {
  int32_t dataSlices = 0;
  int32_t normalizationShift = 0;
  uint32_t activeMask = 0U;
  int32_t firstSelectedOutput = 0;
  int32_t overflowSlices = -1; // < 0: no valid window, the kernel does nothing

  // Lanes 0..7 read the input descriptor and the output window in one load
  // each and broadcast them through the warp.
  CUDA_DEVICE explicit Schedule(const SlicedTensorTmaView &input, const SlicedTensorView &output,
                                const PackedStencilArg &weight) {
    static_assert(sizeof(SliceDescriptor) == 6 * sizeof(uint32_t));
    const int32_t lane = static_cast<int32_t>(threadIdx.x) % WARP_SIZE;
    uint32_t word = 0U;
    if (lane < 6) {
      word = __ldg(reinterpret_cast<const uint32_t *>(input.info) + lane);
    } else if (lane == 6) {
      word = __ldg(reinterpret_cast<const uint32_t *>(&output.info->rH));
    } else if (lane == 7) {
      word = __ldg(reinterpret_cast<const uint32_t *>(&output.info->overflowSlices));
    }
    SliceDescriptor in;
    for (int32_t i = 0; i < 6; ++i) {
      reinterpret_cast<uint32_t *>(&in)[i] = __shfl_sync(0xFFFF'FFFFU, word, i);
    }
    const int32_t selectedOutputRH = static_cast<int32_t>(__shfl_sync(0xFFFF'FFFFU, word, 6));
    const int32_t selectedOverflow = static_cast<int32_t>(__shfl_sync(0xFFFF'FFFFU, word, 7));

    dataSlices = in.rH - in.rL + 1;
    if (dataSlices < 1 || dataSlices >= MAX_SLICES || weight.active_slices() != M || selectedOverflow < 0) {
      return;
    }
    const InputPlanes planes = input_planes(in);
    normalizationShift = planes.shift;
    activeMask = planes.active;
    const int32_t naturalOutputRH = in.rH - normalizationShift + weight.rH - 1;
    if (selectedOutputRH < naturalOutputRH + selectedOverflow) {
      return;
    }
    const int32_t outputRL = selectedOutputRH - dataSlices + 1;
    const int32_t leastOutput = in.rL - normalizationShift + weight.rL - 1;
    firstSelectedOutput = outputRL - 1 - leastOutput;
    overflowSlices = selectedOverflow;
  }

  CUDA_DEVICE CUDA_INLINE bool input_active(int32_t input) const {
    return input < dataSlices && (activeMask & (1U << input)) != 0U;
  }
  CUDA_DEVICE CUDA_INLINE int32_t input_plane(int32_t input) const { return input + 1 - normalizationShift; }
  CUDA_DEVICE CUDA_INLINE int32_t output_plane(int32_t diagonal) const { return diagonal - firstSelectedOutput; }
  CUDA_DEVICE CUDA_INLINE int32_t diagonal_count() const { return dataSlices + M - 1 + overflowSlices; }
  // First active input at or after `input`, or dataSlices.
  CUDA_DEVICE CUDA_INLINE int32_t next_active(int32_t input) const {
    if (input >= dataSlices) {
      return dataSlices;
    }
    const int32_t bit = __ffs(static_cast<int32_t>(activeMask & (~0U << input)));
    return bit == 0 ? dataSlices : bit - 1;
  }
};

// ---------------------------------------------------------------------------
// Preamble (one thread): choose the output digit window of this step, i.e.
// the natural top diagonal plus the overflow digits its carries can need, and
// seed the output descriptor. overflowSlices < 0 marks an invalid window.
// ---------------------------------------------------------------------------
template <int32_t M>
__global__ void preamble(SliceDescriptor *output, PackedStencilArg weight, const SliceDescriptor *input) {
  const SliceDescriptor in = *input;
  const int32_t dataSlices = in.rH - in.rL + 1;
  int32_t overflow = -1;
  InputPlanes planes;
  if (dataSlices >= 1 && dataSlices < MAX_SLICES && weight.active_slices() == M) {
    planes = input_planes(in);
    overflow = overflow_slices(planes, M, weight.positiveSums, weight.negativeSums);
  }
  const int64_t naturalOutputRH = static_cast<int64_t>(in.rH) - planes.shift + weight.rH - 1;
  if (overflow >= 0 && seed_output_descriptor(output, dataSlices, naturalOutputRH + overflow)) {
    output->overflowSlices = overflow;
  } else {
    *output = SliceDescriptor{};
    output->overflowSlices = -1;
  }
}

// ---------------------------------------------------------------------------
// L int32 accumulator tensors; slot k holds diagonal (current + k). The
// tensor's last mode indexes the slots. Emitting a diagonal shifts every slot
// down by one, so all accesses use compile-time slot indices (a dynamically
// indexed ring would make the compiler predicate the MMAs of every slot).
// ---------------------------------------------------------------------------
template <int32_t L, class Tensor> struct AccumulatorFile {
  Tensor storage;

  CUDA_DEVICE CUDA_INLINE void clear() { cute::clear(storage); }
  template <int32_t Slot> CUDA_DEVICE CUDA_INLINE auto slot() {
    return storage(cute::_, cute::_, cute::_, cute::_, cute::Int<Slot>{});
  }
  CUDA_DEVICE CUDA_INLINE auto prototype() { return slot<0>(); }

  // Calls function(accumulator of diagonal current + Offset).
  template <int32_t Offset, class Function> CUDA_DEVICE CUDA_INLINE void visit(Function function) {
    function(slot<Offset>());
  }

  // Splits the current diagonal into its digits (written to `digits`, if
  // given) and its carry, which moves into the next diagonal.
  template <bool StoreDigits, class Digits> CUDA_DEVICE CUDA_INLINE void split(Digits &digits) {
    auto emitted = slot<0>();
#pragma unroll
    for (int32_t element = 0; element < cute::size(emitted); ++element) {
      const int32_t value = emitted(element);
      if constexpr (L == 1) {
        emitted(element) = detail::digit_carry(value);
      } else {
        storage(element + cute::size(emitted)) += detail::digit_carry(value); // slot 1
      }
      if constexpr (StoreDigits) {
        digits(element) = static_cast<core::EmulationType>(value);
      }
    }
    if constexpr (L > 1) {
#pragma unroll
      for (int32_t i = 0; i < cute::size(storage) - cute::size(emitted); ++i) {
        storage(i) = storage(i + cute::size(emitted));
      }
      cute::clear(slot<L - 1>());
    }
  }
  CUDA_DEVICE CUDA_INLINE void propagate() {
    int32_t unused = 0;
    split<false>(unused);
  }
};

// ---------------------------------------------------------------------------
// Ring of shared-memory input tiles filled by TMA.
// ---------------------------------------------------------------------------
template <int32_t Slots, int32_t SlotBytes, int32_t TransactionBytes> struct TmaWindow {
  using Barrier = cutlass::arch::ClusterTransactionBarrier;

  core::EmulationType *shared;
  uint64_t *barriers;
  const CUtensorMap *map;
  uint32_t phases = 0U;

  CUDA_DEVICE TmaWindow(core::EmulationType *sharedStorage, uint64_t *barrierStorage, const CUtensorMap *tensorMap)
      : shared(sharedStorage), barriers(barrierStorage), map(tensorMap) {
    if (threadIdx.x == 0) {
      for (int32_t slot = 0; slot < Slots; ++slot) {
        Barrier::init(barriers + slot, 1);
      }
      // The maps were written through the generic proxy (cudaMemcpy).
      asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(map) : "memory");
    }
    __syncthreads();
  }

  CUDA_DEVICE CUDA_INLINE core::EmulationType *data(int32_t slot) const { return shared + slot * SlotBytes; }

  // Starts loading the box at (x, y, z[, w]) into a free slot.
  template <class... Coordinates> CUDA_DEVICE CUDA_INLINE void issue(int32_t slot, Coordinates... coordinates) {
    static_assert(sizeof...(Coordinates) == 3 || sizeof...(Coordinates) == 4);
    if (threadIdx.x == 0) {
      Barrier::arrive_and_expect_tx(barriers + slot, TransactionBytes);
      const uint32_t destination = static_cast<uint32_t>(__cvta_generic_to_shared(data(slot)));
      const uint32_t barrier = static_cast<uint32_t>(__cvta_generic_to_shared(barriers + slot));
      const int32_t c[4] = {static_cast<int32_t>(coordinates)...};
      if constexpr (sizeof...(Coordinates) == 3) {
        asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, "
                     "%4}], [%5];" ::"r"(destination),
                     "l"(map), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(barrier)
                     : "memory");
      } else {
        asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, "
                     "%4, %5}], [%6];" ::"r"(destination),
                     "l"(map), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]), "r"(barrier)
                     : "memory");
      }
    }
  }

  // Waits until the slot's load has completed (also a CTA barrier).
  CUDA_DEVICE CUDA_INLINE void wait(int32_t slot) {
    if (threadIdx.x == 0) {
      const uint32_t barrier = static_cast<uint32_t>(__cvta_generic_to_shared(barriers + slot));
      asm volatile("{\n\t.reg .pred P;\n\tWAIT_%=:\n\t"
                   "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t@!P bra WAIT_%=;\n\t}" ::"r"(barrier),
                   "r"((phases >> slot) & 1U)
                   : "memory");
    }
    __syncthreads();
    phases ^= 1U << slot;
  }
};

// ---------------------------------------------------------------------------
// The diagonal schedule.
//
// Ping-pong (M <= L + 1): every input is applied with all of its weights in
// one visit. Two slots alternate so the next input loads during arithmetic.
//
// Resident ring (M > L + 1): only L accumulators fit in registers, so each
// input stays resident for R = M - L diagonals. Input i lives in slot i % R
// and is replaced by input i + R after its last weight has been applied.
// The R lowest diagonals (below every stored output plane) are not computed.
// ---------------------------------------------------------------------------
template <int32_t M, int32_t L> struct DiagonalSchedule {
  static constexpr bool PingPong = M <= L + 1;
  static constexpr int32_t Resident = PingPong ? 1 : M - L;
  static constexpr int32_t Slots = PingPong ? 2 : Resident;
  static constexpr int32_t FirstComputedDiagonal = PingPong ? 0 : Resident;

  const Schedule<M> &schedule;
  int32_t nextInput = 0; // ping-pong: next active input not yet issued

  // True when no stored output plane needs arithmetic.
  CUDA_DEVICE CUDA_INLINE bool nothing_to_do() const {
    const int32_t n = schedule.dataSlices;
    const int32_t first = schedule.firstSelectedOutput;
    const int32_t begin = min(n + 1, max(0, FirstComputedDiagonal - first));
    const int32_t end = max(0, min(n + 1, schedule.diagonal_count() - first));
    return begin >= end;
  }

  // First active input of residue class `input % R` at or after `input`.
  CUDA_DEVICE CUDA_INLINE int32_t next_in_class(int32_t input) const {
    for (; input < schedule.dataSlices; input += Resident) {
      if (schedule.input_active(input)) {
        return input;
      }
    }
    return schedule.dataSlices;
  }

  // Fills every slot. Call before staging weights so the loads overlap it.
  template <class Issue> CUDA_DEVICE CUDA_INLINE void prefetch(Issue &&issue) {
    if constexpr (PingPong) {
      nextInput = schedule.next_active(0);
      for (int32_t slot = 0; slot < Slots && nextInput < schedule.dataSlices; ++slot) {
        issue(slot, nextInput);
        nextInput = schedule.next_active(nextInput + 1);
      }
    } else {
      for (int32_t slot = 0; slot < Slots; ++slot) {
        const int32_t input = next_in_class(slot);
        if (input < schedule.dataSlices) {
          issue(slot, input);
        }
      }
    }
  }

  template <class Window, class Issue, class Accumulate, class Emit>
  CUDA_DEVICE CUDA_INLINE void run(Window &window, Issue &&issue, Accumulate &&accumulate, Emit &&emit) {
    const int32_t n = schedule.dataSlices;
    if constexpr (PingPong) {
      int32_t ordinal = 0;
      for (int32_t input = 0; input < n; ++input) {
        const bool active = schedule.input_active(input);
        const int32_t slot = ordinal % 2;
        if (active) {
          window.wait(slot);
          if (ordinal != 0 && nextInput < n) {
            issue(slot ^ 1, nextInput);
            nextInput = schedule.next_active(nextInput + 1);
          }
          accumulate.template operator()<0, 0, L>(slot);
        }
        emit(input);
        if constexpr (L < M) {
          if (active) {
            accumulate.template operator()<M - 1, L - 1, 1>(slot);
          }
        }
        ordinal += active ? 1 : 0;
      }
      for (int32_t diagonal = n; diagonal < schedule.diagonal_count(); ++diagonal) {
        emit(diagonal);
      }
    } else {
      constexpr int32_t R = Resident;
      auto retire = [&](int32_t input) __attribute__((always_inline)) {
        const int32_t next = next_in_class(input + R);
        if (next < n) {
          __syncthreads();
          issue(input % R, next);
        }
      };

      if (schedule.input_active(0)) {
        window.wait(0);
        accumulate.template operator()<R, 0, L>(0);
        retire(0);
      }
      static_for<R - 1>([&](auto k) __attribute__((always_inline)) {
        constexpr int32_t Input = decltype(k)::value + 1;
        if (schedule.input_active(Input)) {
          window.wait(Input);
          accumulate.template operator()<R - Input, 0, L>(Input);
        }
      });
      const bool firstNewActive = schedule.input_active(R);
      if (firstNewActive) {
        window.wait(0);
        accumulate.template operator()<0, 0, 1>(0);
      }
      emit(R);
      if constexpr (L > 1) {
        if (firstNewActive) {
          accumulate.template operator()<1, 0, L - 1>(0);
        }
      }
      if (schedule.input_active(1)) {
        accumulate.template operator()<M - 1, L - 1, 1>(1);
        retire(1);
      }

      // Steady state for diagonal d.
      for (int32_t diagonal = R + 1; diagonal < schedule.diagonal_count(); ++diagonal) {
        static_for<R - 1>([&](auto r) __attribute__((always_inline)) {
          constexpr int32_t Relative = decltype(r)::value;
          const int32_t input = diagonal - R + 1 + Relative;
          if (schedule.input_active(input)) {
            accumulate.template operator()<M - 2 - Relative, L - 1, 1>(input % R);
          }
        });
        const bool newActive = schedule.input_active(diagonal);
        const int32_t newSlot = diagonal % R;
        if (newActive) {
          window.wait(newSlot);
          accumulate.template operator()<0, 0, 1>(newSlot);
        }
        emit(diagonal);
        if constexpr (L > 1) {
          if (newActive) {
            accumulate.template operator()<1, 0, L - 1>(newSlot);
          }
        }
        const int32_t seed = diagonal - R + 1;
        if (schedule.input_active(seed)) {
          accumulate.template operator()<M - 1, L - 1, 1>(seed % R);
          retire(seed);
        }
      }
    }
  }
};

// Metadata computation.
CUDA_DEVICE CUDA_INLINE void publish_metadata(SliceDescriptor *info, uint32_t mask, int32_t mssMin, int32_t mssMax) {
  mask = __reduce_or_sync(0xFFFF'FFFFU, mask);
  mssMin = __reduce_min_sync(0xFFFF'FFFFU, mssMin);
  mssMax = __reduce_max_sync(0xFFFF'FFFFU, mssMax);
  if (threadIdx.x % WARP_SIZE == 0) {
    if (mask != 0U) {
      atomicOr(&info->sMask, mask);
    }
    atomicMin(&info->mssMin, mssMin);
    atomicMax(&info->mssMax, mssMax);
  }
}

// Shared-memory word load by 32-bit shared address.
CUDA_DEVICE CUDA_INLINE uint32_t load_shared_u32(uint32_t address) {
  uint32_t value;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(value) : "r"(address));
  return value;
}

} // namespace ozaki::gpu::kernels

#endif
