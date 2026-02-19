#ifndef HEADER_REFERENCE_SOLVER_HPP
#define HEADER_REFERENCE_SOLVER_HPP

#include <ozaki/core/stencil.hpp>
#include <ozaki/utils/stencil_spec.hpp>
#include <ozaki/utils/tensor.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

using Real = double;

struct ReferenceSolver {
  // Computes the reference stencil integration at the given precision.
  template <typename ComputeType, typename T, size_t... Dims>
  static std::vector<ComputeType> compute(const Tensor<Real> &input, const ozi::Stencil<T, Dims...> &raw_stencil,
                                          int32_t iterations) {
    // Convert input data to ComputeType
    std::vector<ComputeType> u_curr(input.data.size());
    std::transform(input.data.begin(), input.data.end(), u_curr.begin(),
                   [](Real x) { return static_cast<ComputeType>(x); });

    std::vector<ComputeType> u_next = u_curr;

    // Apply stencil iteratively
    constexpr size_t ndims = sizeof...(Dims);
    std::array<size_t, ndims> shape{};
    for (size_t d = 0; d < ndims && d < input.shape.size(); ++d)
      shape[d] = input.shape[d];

    for (int32_t i = 0; i < iterations + 1; ++i) {
      raw_stencil.template apply_to<ComputeType>(u_curr, u_next, shape);
      std::swap(u_curr, u_next);
    }

    return u_curr;
  }
};

#endif // HEADER_REFERENCE_SOLVER_HPP
