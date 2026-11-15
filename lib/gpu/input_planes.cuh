// Which digit planes of a sliced input enter a stencil step, and how many
// extra (overflow) output digits the step's carries can need.
#ifndef OZAKI_GPU_INPUT_PLANES_CUH
#define OZAKI_GPU_INPUT_PLANES_CUH

#include <ozaki/gpu/digit_math.cuh>
#include <ozaki/gpu/sliced_tensor.cuh>

#include <cstdint>

namespace ozaki::gpu {

struct InputPlanes {
  int32_t dataSlices = 0;
  // Planes the window moves down by
  int32_t shift = 0;
  uint32_t active = 0U; // bit i: input i is non-zero
  // Digit range of the most significant input, when known (min <= max).
  int32_t mssMin = 0, mssMax = -1;
};

CUDA_HOST CUDA_DEVICE CUDA_INLINE InputPlanes input_planes(const SliceDescriptor &in) {
  InputPlanes planes;
  const int32_t n = in.rH - in.rL + 1;
  planes.dataSlices = n;
  const uint32_t dataPlanes = n == 31 ? ~1U : (1U << (n + 1)) - 2U; // bits 1..n
  const bool useErrorPlane = in.mssMin == 0 && in.mssMax == 0;
  const uint32_t present = in.sMask & dataPlanes;
  planes.shift = useErrorPlane ? 1 : present == 0U ? n : clz_u32(present) - (31 - n);
  const uint32_t usable = in.sMask & (dataPlanes | (useErrorPlane ? 1U : 0U));
  const int32_t delta = 1 - planes.shift; // physical plane p -> input p - delta
  planes.active = (delta >= 0 ? usable >> delta : usable << -delta) & ((1U << n) - 1U);
  if (planes.shift == 0) {
    planes.mssMin = in.mssMin;
    planes.mssMax = in.mssMax;
  }
  return planes;
}

// Interval arithmetic over the diagonals of one step. Returns how
// many digits the carry out of the top diagonal can still need, or -1.
CUDA_HOST CUDA_DEVICE CUDA_INLINE int32_t overflow_slices(const InputPlanes &planes, int32_t weightSlices,
                                                          const int64_t *positive, const int64_t *negative) {
  const int32_t n = planes.dataSlices;
  int64_t carryMin = 0, carryMax = 0;
  for (int32_t diagonal = 0; diagonal < n + weightSlices - 1; ++diagonal) {
    int64_t low = carryMin, high = carryMax;
    for (int32_t weight = 0; weight < weightSlices; ++weight) {
      const int32_t input = diagonal - weight;
      if (input < 0 || input >= n || (planes.active & (1U << input)) == 0U) {
        continue;
      }
      const bool top = input == n - 1 && planes.mssMin <= planes.mssMax;
      const int64_t digitMin = top ? planes.mssMin : -128;
      const int64_t digitMax = top ? planes.mssMax : 127;
      low += digitMin * positive[weight] + digitMax * negative[weight];
      high += digitMax * positive[weight] + digitMin * negative[weight];
    }
    carryMin = extract_carry(low, static_cast<int8_t>(low));
    carryMax = extract_carry(high, static_cast<int8_t>(high));
  }
  int32_t overflow = 0;
  for (; carryMin != 0 || carryMax != 0; ++overflow) {
    if (overflow == 16) {
      return -1;
    }
    carryMin = extract_carry(carryMin, static_cast<int8_t>(carryMin));
    carryMax = extract_carry(carryMax, static_cast<int8_t>(carryMax));
  }
  return overflow;
}

} // namespace ozaki::gpu

#endif // OZAKI_GPU_INPUT_PLANES_CUH
