#ifndef OZAKI_ZERO_BOUNDARY_HPP
#define OZAKI_ZERO_BOUNDARY_HPP

#include "tensor.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>

template <typename T> Tensor<T> zero_boundary_copy(const Tensor<T> &input, size_t halo) {
  if (input.shape.size() != 2U && input.shape.size() != 3U) {
    throw std::invalid_argument("Zero-boundary stencil integration supports only 2D or 3D tensors.");
  }
  for (const size_t extent : input.shape) {
    if (extent <= 2 * halo) {
      throw std::invalid_argument("Tensor has no nonempty stencil interior.");
    }
  }

  Tensor<T> result = input;
  size_t volume = 1;
  for (const size_t extent : input.shape) {
    if (volume > std::numeric_limits<size_t>::max() / extent) {
      throw std::overflow_error("Tensor volume overflows size_t.");
    }
    volume *= extent;
  }
  if (result.data.size() != volume) {
    throw std::invalid_argument("Tensor shape does not match its storage size.");
  }
  for (size_t linear = 0; linear < volume; ++linear) {
    size_t remainder = linear;
    bool boundary = false;
    for (size_t dimension = input.shape.size(); dimension-- > 0;) {
      const size_t coordinate = remainder % input.shape[dimension];
      remainder /= input.shape[dimension];
      boundary = boundary || coordinate < halo || coordinate >= input.shape[dimension] - halo;
    }
    if (boundary) {
      result.data[linear] = static_cast<T>(0);
    }
  }
  return result;
}

#endif
