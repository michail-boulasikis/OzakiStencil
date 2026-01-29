#ifndef HEADER_OZI_CUH
#define HEADER_OZI_CUH

#include "../core/ozconcepts.hpp"
#include "../core/stencil.hpp"
#include "gpu_checks.cuh"

#include <concepts>
#include <cuda_fp16.h>
#include <limits>
#include <type_traits>

#ifndef RDC_THREADS_PER_BLOCK
#define RDC_THREADS_PER_BLOCK 512
#endif

namespace ozigpu {

template <std::floating_point T>
  requires(sizeof(T) > 1)
struct OzakiFP_Int8 {
  using data_type = T;
  using target_type = int8_t;
  using acc_type = int32_t;
};

template <ozi::OzakiPair O> struct alignas(256) OzakiDescriptor {
  using Data = typename O::data_type;
  using Target = typename O::target_type;
  using Acc = typename O::acc_type;

  int32_t e_max = std::numeric_limits<int32_t>::min();
  int32_t e_min = std::numeric_limits<int32_t>::max();
  int32_t n_slices_input = 8;
  int32_t n_slices_stencil = 1;

  int32_t n_shifts = 0;
  int32_t shift_required = 0;
  int32_t just_shifted = 1;
  Target overflow_danger_threshold = std::numeric_limits<Target>::max() / 2;
  int32_t carry_slice_index = 0;
};

namespace detail {

// Vectorized load/store helper
template <typename T> struct FourValues {
  T a[4];
};

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
#if __CUDA_ARCH__ >= 100
template <> struct FloatTraits<__float128> {
  static constexpr int32_t MantissaBits = 112;
  static constexpr int32_t ExponentBias = 16383;
};
#endif

template <typename To, typename From> __device__ __forceinline__ To bit_cast(const From &src) {
  static_assert(sizeof(To) == sizeof(From), "Size mismatch in bit_cast");
  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

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

} // namespace detail
} // namespace ozigpu

namespace ozi {
template <std::floating_point T> struct OzakiPairTraits<ozigpu::OzakiFP_Int8<T>> {
  using Data = typename ozigpu::OzakiFP_Int8<T>::data_type;
  using Target = typename ozigpu::OzakiFP_Int8<T>::target_type;
  using Acc = typename ozigpu::OzakiFP_Int8<T>::acc_type;

  __device__ __forceinline__ static Target down_cast(Data a, int32_t scale) {
    if constexpr (std::is_same_v<Data, double>) {
      double a_scaled = scalbn(a, scale);
      int32_t temp;
      asm("cvt.rni.sat.s8.f64 %0, %1;" : "=r"(temp) : "d"(a_scaled));
      return static_cast<Target>(temp);
    } else if constexpr (std::is_same_v<Data, float>) {
      float a_scaled = scalbn(static_cast<float>(a), scale);
      int32_t temp;
      asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(temp) : "f"(a_scaled));
      return static_cast<Target>(temp);
    } else {
      static_assert(false, "down_cast not implemented for this type yet.");
    }
  }

  __host__ __device__ __forceinline__ static Data up_cast(Target a, int32_t scale) {
    return OzakiTraits<Data>::scalbn(static_cast<Data>(a), -scale);
  }

  __host__ __device__ __forceinline__ static Data up_cast_acc(Acc a, int32_t scale, int32_t n_mult) {
    return OzakiTraits<Data>::scalbn(static_cast<Data>(a), -(n_mult * OzakiTraits<Target>::scaling_factor(scale)));
  }

  // Stub matmul implementation (not used in stencil code but required by concept)
  static void matmul(Acc *C, const Target *A, const Target *B, size_t m, size_t n, size_t k) {
    // Not implemented for stencil operations
  }
};

} // namespace ozi

namespace ozigpu {
using namespace ozi;

template <OzakiPair O>
__host__ typename O::data_type total_integration_scale_factor(typename O::data_type a, int64_t time_steps,
                                                              int64_t shifts) {
  using Data = typename O::data_type;
  using Target = typename O::target_type;

  int32_t exp;
  double frac = OzakiTraits<double>::frexp(static_cast<double>(a), &exp);
  return static_cast<Data>(
      OzakiTraits<double>::scalbn(pow(frac, time_steps), time_steps * exp + (CHAR_BIT * sizeof(Target) - 1) * shifts));
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

/** Finds the dynamic range [pos(MSB), pos(LSB)] of a set of numbers */
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
  while (4 * i < n) {
    const Data *src = in + 4 * i;
    detail::FourValues<Data> vals = *reinterpret_cast<const detail::FourValues<Data> *>(src);

    Data abs0 = OzakiTraits<Data>::abs(vals.a[0]);
    Data abs1 = OzakiTraits<Data>::abs(vals.a[1]);
    Data abs2 = OzakiTraits<Data>::abs(vals.a[2]);
    Data abs3 = OzakiTraits<Data>::abs(vals.a[3]);

    local_max_val = OzakiTraits<Data>::max(
        OzakiTraits<Data>::max(OzakiTraits<Data>::max(OzakiTraits<Data>::max(abs0, abs1), abs2), abs3), local_max_val);

    local_min_val = detail::dyn_min_lsb(
        vals.a[0],
        detail::dyn_min_lsb(vals.a[1], detail::dyn_min_lsb(vals.a[2], detail::dyn_min_lsb(vals.a[3], local_min_val))));
    i += GRID_SIZE;
  }

  constexpr uint32_t FULL_MASK = 0xFFFF'FFFF;
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset /= 2) {
    local_max_val = OzakiTraits<Data>::max(local_max_val, __shfl_down_sync(FULL_MASK, local_max_val, offset));
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
      local_max_val = OzakiTraits<Data>::max(local_max_val, __shfl_down_sync(mask, local_max_val, offset));
      local_min_val = min(local_min_val, __shfl_down_sync(mask, local_min_val, offset));
    }
    if (LANE_ID == 0) {
      int32_t block_partial_result = static_cast<int32_t>(ceil(log2(local_max_val)));
      atomicMax(&out->e_max, block_partial_result);
      atomicMin(&out->e_min, local_min_val);
    }
  }
}

/** Computes the Ozaki decomposition of a set of numbers and stores it in *out */
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

  if (desc->n_slices_input == 0) {
    total_slices = 2 + (desc->e_max - desc->e_min) / (CHAR_BIT * sizeof(Target) - 1);
    if (tid == 0)
      desc->n_slices_input = total_slices;
  }

  detail::FourValues<Data> ins = reinterpret_cast<const detail::FourValues<Data> *>(in)[tid];
  detail::FourValues<Target> outs{{0, 0, 0, 0}};

#pragma unroll 2
  for (int32_t s = 0; s < total_slices; ++s) {
    const size_t flat_offset = static_cast<size_t>(s) * (n / 4) + static_cast<size_t>(tid);
    if (s != 0) {
      int32_t shift_amount = OzakiTraits<Target>::scaling_factor(emax);
      emax = OzakiTraits<Target>::slice_step(emax);

#pragma unroll
      for (int32_t v = 0; v < 4; ++v) {
        const Target v_converted = OzakiPairTraits<O>::down_cast(ins.a[v], shift_amount);
        ins.a[v] -= OzakiPairTraits<O>::up_cast(v_converted, shift_amount);
        outs.a[v] = v_converted;
      }
    }
    reinterpret_cast<detail::FourValues<Target> *>(out)[flat_offset] = outs;
  }
}

/** Reconstructs (mends) the slices of an Ozaki decomposition */
template <OzakiPair O, typename OutputType = typename O::data_type>
__global__ void mend(OutputType *__restrict__ out, const typename O::target_type *__restrict__ in,
                     OzakiDescriptor<O> *__restrict__ desc, const size_t n) {
  using Data = typename O::data_type;
  using Target = typename O::target_type;

  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n / 4)
    return;

  int32_t n_shifts = desc->n_shifts;
  int32_t global_max_exp = desc->e_max + 1 * (CHAR_BIT * sizeof(Target) - 1);
  int32_t total_slices = desc->n_slices_input;
  int32_t carry_idx = desc->carry_slice_index;

  constexpr int32_t SLICE_BITS = CHAR_BIT * sizeof(Target) - 1;

  detail::FourValues<OutputType> outs{{0, 0, 0, 0}};
  detail::FourValues<Target> ins{};

  for (int32_t logic_s = total_slices - 1; logic_s >= 0; --logic_s) {
    int32_t current_slice_exp = global_max_exp - (logic_s * SLICE_BITS);
    int32_t shift_amount = SLICE_BITS - current_slice_exp;

    const int32_t physical_slice = (carry_idx + logic_s) % total_slices;
    const size_t flat_offset = physical_slice * (n / 4) + static_cast<size_t>(tid);
    ins = reinterpret_cast<const detail::FourValues<Target> *>(in)[flat_offset];

#pragma unroll
    for (int32_t v = 0; v < 4; ++v) {
      outs.a[v] += static_cast<OutputType>(OzakiPairTraits<O>::up_cast(ins.a[v], shift_amount));
    }
  }
  reinterpret_cast<detail::FourValues<OutputType> *>(out)[tid] = outs;
}

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
  __syncthreads();

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

/** State update kernel for integration */
template <OzakiPair O> __global__ void update_ozaki_desc(OzakiDescriptor<O> *__restrict__ desc) {
  if (threadIdx.x == 0) {
    int32_t perform_shift = desc->shift_required;
    if (perform_shift) {
      int32_t total = desc->n_slices_input;
      desc->carry_slice_index = (desc->carry_slice_index + total - 1) % total;
      desc->n_shifts += 1;
      desc->shift_required = 0;
    }
    desc->just_shifted = perform_shift;
  }
}

} // namespace ozigpu

#endif // HEADER_OZI_CUH
