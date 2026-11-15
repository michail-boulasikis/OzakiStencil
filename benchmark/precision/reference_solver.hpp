#ifndef HEADER_REFERENCE_SOLVER_HPP
#define HEADER_REFERENCE_SOLVER_HPP

#include <ozaki/core/stencil.hpp>
#include <ozaki/utils/stencil_spec.hpp>
#include <ozaki/utils/tensor.hpp>
#include <ozaki/utils/zero_boundary.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

struct ReferenceSolver {
  // Computes the reference stencil integration at the given precision.
  template <typename ComputeType, typename T, size_t... Dims>
  static std::vector<ComputeType> compute(const Tensor<__float128> &input, const ozaki::Stencil<T, Dims...> &raw_stencil,
                                          int32_t iterations) {
    // Homogeneous Dirichlet boundary, as in StencilIntegrator: the halo is
    // zero in both ping-pong buffers and apply_to writes only interior cells.
    const Tensor<__float128> bounded = zero_boundary_copy(input, raw_stencil.shape[0] / 2);
    std::vector<ComputeType> u_curr(bounded.data.size());
    std::transform(bounded.data.begin(), bounded.data.end(), u_curr.begin(),
                   [](__float128 x) { return static_cast<ComputeType>(x); });

    std::vector<ComputeType> u_next = u_curr;

    // Apply stencil iteratively
    constexpr size_t ndims = sizeof...(Dims);
    if (ndims != input.shape.size())
      throw std::invalid_argument("Stencil has " + std::to_string(ndims) + " dimensions but input tensor has " +
                                  std::to_string(input.shape.size()) + " dimensions");
    std::array<size_t, ndims> shape{};
    for (size_t d = 0; d < ndims && d < input.shape.size(); ++d)
      shape[d] = input.shape[d];

    for (int32_t i = 0; i < iterations; ++i) {
      raw_stencil.template apply_to<ComputeType>(u_next, u_curr, shape);
      std::swap(u_curr, u_next);
    }

    return u_curr;
  }
};

#endif // HEADER_REFERENCE_SOLVER_HPP
