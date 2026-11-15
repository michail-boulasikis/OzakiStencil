// Register layouts of the stencil coefficients as the B operand of
// mma.m16n8k32.row.col.s8
// Coefficient slices are indexed by significance: slice 0 is the most
// significant, as produced by cpu::slice_coefficients.
#ifndef OZAKI_GPU_WEIGHT_LAYOUT_HPP
#define OZAKI_GPU_WEIGHT_LAYOUT_HPP

#include <ozaki/gpu/digit_math.cuh>

#include <cstdint>

namespace ozaki::gpu {

// 2D stencils of side 3, 7 or 15.
struct Mma2dWeightLayout {
  int32_t side;
  int32_t slices;

  static constexpr int32_t TESS = 2;
  static constexpr int32_t WORDS = 2;
  static constexpr int32_t LANES = 32;

  CUDA_HOST CUDA_DEVICE constexpr int32_t group_cols() const { return side < 8 ? 8 : 16; } // P
  CUDA_HOST CUDA_DEVICE constexpr int32_t k_blocks() const { return cdiv(side * group_cols(), 32); }
  CUDA_HOST CUDA_DEVICE constexpr int32_t n8_per_slice() const { return group_cols() / 8; }
  CUDA_HOST CUDA_DEVICE constexpr int32_t n8() const { return slices * n8_per_slice(); }
  CUDA_HOST CUDA_DEVICE constexpr int32_t bytes() const { return TESS * k_blocks() * n8() * WORDS * LANES * 4; }

  CUDA_HOST CUDA_DEVICE constexpr int32_t word_index(int32_t tess, int32_t k_block, int32_t n8_index, int32_t word,
                                                     int32_t lane) const {
    return (((tess * k_blocks() + k_block) * n8() + n8_index) * WORDS + word) * LANES + lane;
  }

  // Byte of logical B element (tessellation, K row, N column).
  CUDA_HOST CUDA_DEVICE constexpr int32_t byte_index(int32_t tess, int32_t k_row, int32_t col) const {
    const int32_t k_in_block = k_row % 32;
    const int32_t lane = (col % 8) * 4 + (k_in_block % 16) / 4;
    return word_index(tess, k_row / 32, col / 8, k_in_block / 16, lane) * 4 + k_in_block % 4;
  }
};

// 3D stencils, packed as 5x5x5 (a 3x3x3 stencil fills the low corner).
struct Mma3dWeightLayout {
  int32_t slices;

  static constexpr int32_t COEFFICIENT_Z = 5;
  static constexpr int32_t TESS = 2;
  static constexpr int32_t WORDS = 2;
  static constexpr int32_t LANES = 32;
  static constexpr int32_t OUTPUTS = 6;

  struct KCoordinate {
    int32_t tess;
    int32_t k;
  };

  CUDA_HOST CUDA_DEVICE static constexpr KCoordinate k_coordinate(int32_t y, int32_t input_col) {
    if (input_col <= 3) {
      return {0, 4 * y + input_col};
    }
    if (input_col <= 5) {
      return {0, 4 * (5 + y / 2) + 2 * (y % 2) + input_col - 4};
    }
    return {1, 4 * y + input_col - 6};
  }

  CUDA_HOST CUDA_DEVICE constexpr int32_t bytes() const { return COEFFICIENT_Z * TESS * slices * WORDS * LANES * 4; }

  CUDA_HOST CUDA_DEVICE constexpr int32_t word_index(int32_t z, int32_t tess, int32_t slice, int32_t word,
                                                     int32_t lane) const {
    return (((z * TESS + tess) * slices + slice) * WORDS + word) * LANES + lane;
  }

  CUDA_HOST CUDA_DEVICE constexpr int32_t byte_index(int32_t z, int32_t tess, int32_t slice, int32_t k,
                                                     int32_t output) const {
    return word_index(z, tess, slice, k / 16, 4 * output + (k % 16) / 4) * 4 + k % 4;
  }
};

} // namespace ozaki::gpu

#endif // OZAKI_GPU_WEIGHT_LAYOUT_HPP
