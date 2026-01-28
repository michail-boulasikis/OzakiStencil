#ifndef HEADER_OZI_CUH
#define HEADER_OZI_CUH

#include "../core/ozconcepts.hpp"
#include "../core/stencil.hpp"
#include "gpu_checks.cuh"

#include <concepts>
#include <device_atomic_functions.h>
#include <limits>
#include <type_traits>

#ifndef RDC_THREADS_PER_BLOCK
#define RDC_THREADS_PER_BLOCK 512
#endif

namespace ozigpu {

using namespace ozi;

#ifdef __CUDACC__
// for vectorized load store ops
template <typename T> struct FourValues {
  T a[4];
};

template <OzakiPair O> __host__ __device__ struct alignas(256) OzakiDescriptor {
  using Data = typename O::data_type;
  using Target = typename O::target_type;
  using Acc = typename O::acc_type;

  int32_t e_max = std::numeric_limits<int32_t>::min(); //< Maximum exponent bit represented by the slices without carry.
  int32_t e_min = std::numeric_limits<int32_t>::max(); //< Minimum exponent bit represented by the slices without carry.
  int32_t n_slices_input = 8;                          //< Number of slices of the input.
  int32_t n_slices_stencil = 1;                        //< Number of slices of the stencil.

  int32_t n_shifts = 0;       //< Total number of shifts done during the stencil loop.
  int32_t shift_required = 0; //< Whether we need to shift the slices to incorporate the carry slice and throw away the
                              // least significant slice.
  int32_t just_shifted = 1;   //< Whether a window shift just occurred. Used to conceptually zero the buffer.
  Target overflow_danger_threshold = std::numeric_limits<Target>::max() /
                                     2; //< If any |element| of the carry slice is
                                        // higher than this we risk overflow next iteration.
                                        // A simple way to guard against overflow - might make it more complex later.
  int32_t carry_slice_index = 0;        //< The index of the carry slice.
};

template <typename To, typename From> __device__ __forceinline__ To bit_cast(const From &src) {
  static_assert(sizeof(To) == sizeof(From), "Size mismatch in bit_cast");
  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <OzakiPair O>
__host__ __device__ typename O::data_type total_integration_scale_factor(typename O::data_type a, int64_t time_steps,
                                                                         int64_t shifts) {
  using std::frexp;
  using std::pow;
  using std::scalbn;
  using Data = typename O::data_type;
  using Target = typename O::target_type;

  int32_t exp;
  double frac;
  frac = static_cast<double>(frexp(a, &exp));
  return static_cast<Data>(scalbn(pow(frac, time_steps), time_steps * exp + (CHAR_BIT * sizeof(Target) - 1) * shifts));
}

template <size_t Bytes> struct UintByType;
template <> struct UintByType<2> {
  using type = uint16_t;
};
template <> struct UintByType<4> {
  using type = uint32_t;
};
template <> struct UintByType<8> {
  using type = uint64_t;
};

template <typename T> using RawInt = typename UintByType<sizeof(T)>::type;

template <typename T> struct FloatTraits;
template <> struct FloatTraits<__half> {
  static constexpr int32_t MantissaBits = 10;
  static constexpr int32_t ExponentBias = 15;
};
template <> struct FloatTraits<float> {
  static constexpr int32_t MantissaBits = 23;
  static constexpr int32_t ExponentBias = 127;
};
template <> struct FloatTraits<double> {
  static constexpr int32_t MantissaBits = 52;
  static constexpr int32_t ExponentBias = 1023;
};
template <> struct FloatTraits<__float128> {
  static constexpr int32_t MantissaBits = 112;
  static constexpr int32_t ExponentBias = 16383;
};

template <typename T> __device__ __forceinline__ int32_t ffs(T val) {
  if constexpr (sizeof(T) <= 4) {
    return __ffs(static_cast<int32_t>(val));
  } else if constexpr (sizeof(T) <= 8) {
    return __ffsll(static_cast<int64_t>(val));
  } else {
    auto lo = static_cast<uint64_t>(val);
    if (lo != 0)
      return __ffsll(lo);
    auto hi = static_cast<uint64_t>(val >> 64);
    int hi_bit = __ffsll(hi);
    return hi_bit ? hi_bit + 64 : 0;
  }
}

/** Returns min(first_set_lsb(a), current_lsb) */
template <typename T> __device__ __forceinline__ int32_t dyn_min_lsb(T a, int32_t current_lsb) {
  using IntType = RawInt<T>;
  using Traits = FloatTraits<T>;

  if (a == static_cast<T>(0)) // zero does not count.
    return current_lsb;

  IntType raw = bit_cast<IntType>(a);

  constexpr IntType One = 1;
  constexpr int32_t M_Bits = Traits::MantissaBits;
  constexpr IntType M_Mask = (One << M_Bits) - 1;
  constexpr int32_t TotalBits = sizeof(T) * CHAR_BIT;
  IntType exponent_bits = (raw >> M_Bits) & ((One << (TotalBits - 1 - M_Bits)) - 1);
  IntType mantissa = raw & M_Mask;
  if (exponent_bits != 0) {
    mantissa |= (One << M_Bits);
  } else {
    exponent_bits = 1;
  }
  int32_t ffs_index = ffs(mantissa);
  int32_t bit_exponent = (static_cast<int32_t>(exponent_bits) - Traits::ExponentBias) - (M_Bits - (ffs_index - 1));
  return min(current_lsb, bit_exponent);
}

template <std::integral T> __host__ __device__ __forceinline__ int32_t ozaki_scaling_factor(int32_t *v) {
  int32_t factor = (CHAR_BIT * sizeof(T) - 1) - *v;
  *v -= CHAR_BIT * sizeof(T) - 1;
  return factor;
}

template <std::floating_point T>
  requires(sizeof(T) > 1)
struct OzakiFP_Int8 {
  using data_type = T;
  using target_type = int8_t;
  using acc_type = int32_t;
};

__device__ __forceinline__ int8_t down_cast(double a, int32_t scale) {
  double a_scaled = scalbn(a, scale);
  int32_t temp;
  asm("cvt.rni.sat.s8.f64 %0, %1;" : "=r"(temp) : "d"(a_scaled));
  return static_cast<int8_t>(temp);
}
__device__ __forceinline__ int8_t down_cast(float a, int32_t scale) {
  float a_scaled = scalbn(a, scale);
  int32_t temp;
  asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(temp) : "f"(a_scaled));
  return static_cast<int8_t>(temp);
}

template <std::floating_point T, std::integral U> __device__ __forceinline__ T up_cast(U a, int32_t scale) {
  return scalbn(static_cast<T>(a), -scale);
}

template <typename T> __global__ void fill(T *__restrict__ in, const T val, size_t n) {
  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n)
    in[tid] = val;
}

template <typename T> __global__ void scale(T *__restrict__ in, const double val, size_t n) {
  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n)
    in[tid] = static_cast<T>(static_cast<double>(in[tid]) * val);
}

/** Finds the dynamic range [pos(MSB), pos(LSB)] of a set of numbers. pos(MSB) - pos(LSB) + 1 gives the number of bits
 * needed to represent the given set of numbers using a fixed point representation without loss of information. */
template <OzakiPair O>
__global__ void dyn_range(OzakiDescriptor<O> *__restrict__ out, const typename O::data_type *__restrict__ in,
                          size_t n) {
  using Data = typename O::data_type;

  constexpr uint32_t WARPS_PER_BLOCK = RDC_THREADS_PER_BLOCK / WARP_SIZE;
  constexpr uint32_t BLOCK_STRIDE = RDC_THREADS_PER_BLOCK;
  __shared__ Data smem_max[WARPS_PER_BLOCK];
  __shared__ int32_t smem_min[WARPS_PER_BLOCK];

  const uint32_t GRID_SIZE = BLOCK_STRIDE * gridDim.x;
  size_t i = threadIdx.x + blockIdx.x * BLOCK_STRIDE;

  Data local_max_val = static_cast<Data>(0);
  int32_t local_min_val = std::numeric_limits<int32_t>::max();
#pragma unroll 4
  while (4 * i < n) { // I assume that n is a multiple of four.
    const Data *src = in + 4 * i;
    FourValues<Data> vals = *reinterpret_cast<const FourValues<Data> *>(src);
    local_max_val = max(max(max(max(abs(vals.a[0]), abs(vals.a[1])), abs(vals.a[2])), abs(vals.a[3])), local_max_val);
    local_min_val =
        dyn_min_lsb(vals.a[0], dyn_min_lsb(vals.a[1], dyn_min_lsb(vals.a[2], dyn_min_lsb(vals.a[3], local_min_val))));
    i += GRID_SIZE;
  }

  constexpr uint32_t FULL_MASK = 0xFFFF'FFFF;
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset /= 2) {
    local_max_val = max(local_max_val, __shfl_down_sync(FULL_MASK, local_max_val, offset));
    local_min_val = min(local_min_val, __shfl_down_sync(FULL_MASK, local_min_val, offset));
  }

  const uint32_t WARP_ID = threadIdx.x / WARP_SIZE;
  const uint32_t LANE_ID = threadIdx.x % WARP_SIZE;

  if (LANE_ID == 0) {
    smem_max[WARP_ID] = local_max_val;
    smem_min[WARP_ID] = local_min_val;
  }
  __syncthreads();

  if ((WARP_ID == 0) && (LANE_ID < WARPS_PER_BLOCK)) {
    uint32_t mask = __activemask();
    local_max_val = smem_max[LANE_ID];
    local_min_val = smem_min[LANE_ID];
#pragma unroll
    for (uint32_t offset = WARPS_PER_BLOCK / 2; offset > 0; offset /= 2) {
      local_max_val = max(local_max_val, __shfl_down_sync(mask, local_max_val, offset));
      local_min_val = min(local_min_val, __shfl_down_sync(mask, local_min_val, offset));
    }
    if (LANE_ID == 0) {
      int32_t block_partial_result = static_cast<int32_t>(floor(log2(local_max_val))) + 1;
      atomicMax(&out->e_max, block_partial_result);
      atomicMin(&out->e_min, local_min_val);
    }
  }
}

/** Computes the Ozaki decomposition of a set of numbers and stores it in *out. The carry slice, which is assumed to be
 * at exactly &out, gets assigned zeroes. */
template <OzakiPair O>
__global__ void slice(typename O::target_type *__restrict__ out, const typename O::data_type *__restrict__ in,
                      OzakiDescriptor<O> *__restrict__ desc, const size_t n) {
  using Data = typename O::data_type;
  using Target = typename O::target_type;
  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n / 4)
    return;

  int32_t emax = desc->e_max;
  int32_t total_slices = desc->n_slices_input;
  // We interpret a zero as a request to dynamically determine the amount of slices
  // Everything that follows assumes that adequate amount of space was allocated.
  if (desc->n_slices_input == 0) {
    total_slices = 2 + (desc->e_max - desc->e_min) / (CHAR_BIT * sizeof(Target) - 1);
    if (tid == 0)
      desc->n_slices_input = total_slices;
  }

  FourValues<Data> ins = reinterpret_cast<const FourValues<Data> *>(in)[tid];
  FourValues<Target> outs{{0, 0, 0, 0}};

#pragma unroll 2
  for (int32_t s = 0; s < total_slices; ++s) {
    const size_t flat_offset = static_cast<size_t>(s) * (n / 4) + static_cast<size_t>(tid);
    if (s != 0) { // s == 0 is carry slice in the beginning, all zeros no need to compute
      int32_t shift_amount = ozaki_scaling_factor<Target>(&emax);
#pragma unroll
      for (int32_t v = 0; v < 4; ++v) {
        const Target v_converted = down_cast(ins.a[v], shift_amount);
        ins.a[v] -= up_cast<Data, Target>(v_converted, shift_amount);
        outs.a[v] = v_converted;
      }
    }
    reinterpret_cast<FourValues<Target> *>(out)[flat_offset] = outs;
  }
}

/** Reconstructs (mends) the slices of an Ozaki decomposition of numbers to the original values.
 * The exponents are not assumed to be monotonic, but shuffled due to integrating, so the indices are computed. */
template <OzakiPair O, typename OutputType = typename O::data_type>
__global__ void mend(OutputType *__restrict__ out, const typename O::target_type *__restrict__ in,
                     OzakiDescriptor<O> *__restrict__ desc, const size_t n) {
  using Data = typename O::data_type;
  using Target = typename O::target_type;

  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n / 4)
    return;

  int32_t n_shifts = desc->n_shifts;
  int32_t global_max_exp = desc->e_max + (/*n_shifts +*/ 1) * (CHAR_BIT * sizeof(Target) - 1);
  int32_t total_slices = desc->n_slices_input;
  int32_t carry_idx = desc->carry_slice_index;

  constexpr int32_t SLICE_BITS = CHAR_BIT * sizeof(Target) - 1;

  FourValues<OutputType> outs{{0, 0, 0, 0}};
  FourValues<Target> ins{};

  for (int32_t logic_s = total_slices - 1; logic_s >= 0; --logic_s) {

    // Calculate exponent for this specific slice manually
    // Original logic: factor = bits - current_v; current_v -= bits;
    // So current_v for slice 's' is: global_max_exp - s * SLICE_BITS
    int32_t current_slice_exp = global_max_exp - (logic_s * SLICE_BITS);

    // The scaling factor required by up_cast/scalbn
    int32_t shift_amount = SLICE_BITS - current_slice_exp;

    // Map to physical slice
    const int32_t physical_slice = (carry_idx + logic_s) % total_slices;
    const size_t flat_offset = physical_slice * (n / 4) + static_cast<size_t>(tid);
    ins = reinterpret_cast<const FourValues<Target> *>(in)[flat_offset];

    for (int32_t v = 0; v < 4; ++v) {
      outs.a[v] += static_cast<OutputType>(up_cast<Data, Target>(ins.a[v], shift_amount));
    }
  }
  reinterpret_cast<FourValues<OutputType> *>(out)[tid] = outs;
}
/** Computes S @ u, where u is the ozaki decomposition of a tensor, and S is a stencil. Boundaries are excluded in the
 * calculation. Also detects whether an overflow is possible on the carry slice and signals a slice shift if this is the
 * case. */
template <OzakiPair O, size_t... Dims>
  requires(sizeof...(Dims) <= 3) && std::is_integral_v<typename O::target_type>
__global__ void stencil_v1(typename O::target_type *__restrict__ out, const typename O::target_type *__restrict__ in,
                           const Stencil<typename O::target_type, Dims...> s,
                           const typename Stencil<typename O::target_type, Dims...>::shape_type input_shape,
                           OzakiDescriptor<O> *__restrict__ desc) {
  using Data = typename O::data_type;
  using Target = typename O::target_type;
  using UTarget = std::make_unsigned_t<Target>;
  using Acc = typename O::acc_type;

  // loads from the descriptor.
  const int32_t total_slices = desc->n_slices_input;
  const int32_t carry_idx = desc->carry_slice_index;
  const Target threshold = desc->overflow_danger_threshold;
  // used for the reduction.
  __shared__ int32_t smem[32];

  int32_t i, j, k;
  i = threadIdx.x + blockDim.x * blockIdx.x;
  if constexpr (sizeof...(Dims) > 1)
    j = threadIdx.y + blockDim.y * blockIdx.y;
  if constexpr (sizeof...(Dims) > 2)
    k = threadIdx.z + blockDim.z * blockIdx.z;
  int32_t tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

  // zero smem
  if (tid < 32) {
    smem[tid] = 0;
  }
  __syncthreads();

  // Bounds check
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

  // computing constants used for indexing.
  size_t slice_stride = 1;
  for (auto d : input_shape) {
    slice_stride *= d;
  }
  // This represents the integer growth factor per step
  Acc stencil_gain = 0;
  for (const auto &c : s.coeffs) {
    stencil_gain += abs(static_cast<Acc>(c));
  }
  const auto input_strides = ozi::make_strides_from_shape(input_shape);

  Acc local_val = static_cast<Acc>(0);

  // process all slices (Logical N-1 down to 0)
  // logical N-1 = least significant slice (LSS)
  // logical 0 = carry slice (Holds MSBs, must be stenciled)
  if (valid_thread) {
    size_t out_offset = 0;
    if constexpr (sizeof...(Dims) == 1)
      out_offset = i;
    else if constexpr (sizeof...(Dims) == 2)
      out_offset = i + j * input_shape[0];
    else if constexpr (sizeof...(Dims) == 3)
      out_offset = i + j * input_shape[1] + k * input_shape[0] * input_shape[1];

#pragma unroll
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
  // check whether the carry slice risks overflow (block reduction on boolean)
  int32_t dangerous = __any_sync(__activemask(), (abs(local_val) > static_cast<Acc>(threshold)));
  if (tid % WARP_SIZE == 0) // lane 0 writes result
    smem[tid / WARP_SIZE] = dangerous;
  __syncthreads(); // wait for shared memory update
  // First warp does the final reduction and write
  if (tid < 32) {
    int32_t flag = smem[tid];
    flag = __any_sync(__activemask(), flag);
    // elect a leader thread to write on the flag
    if (flag && (__ffs(__activemask()) - 1 == tid)) {
      atomicExch(&desc->shift_required, 1);
    }
  }
}

/** State update kernel for integration */
template <OzakiPair O> __global__ void update_ozaki_desc(OzakiDescriptor<O> *__restrict__ desc) {
  if (threadIdx.x == 0) {
    // see whether a shift is required by previous stencil step
    int32_t perform_shift = desc->shift_required;
    if (perform_shift) {
      // 1. update the carry index (backwards shift logic, LSS becomes new carry)
      int32_t total = desc->n_slices_input;
      desc->carry_slice_index = (desc->carry_slice_index + total - 1) % total;
      // 2. track number of shifts for final mending.
      desc->n_shifts += 1;
      // 3. clear the required shift
      desc->shift_required = 0;
    }
    // 4. set the flag for the UPCOMING stencil step
    // If we shifted, the next stencil invokation must treat the new carry slice as 0.
    // If we didn't, it proceeds as normal.
    desc->just_shifted = perform_shift;
  }
}

#endif
} // namespace ozigpu

#endif // HEADER_OZI_CUH
