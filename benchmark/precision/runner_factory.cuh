#ifndef HEADER_RUNNER_FACTORY_CUH
#define HEADER_RUNNER_FACTORY_CUH

#include "ozaki_runner.cuh"

#include <ozaki/utils/stencil_registry.hpp>
#include <ozaki/utils/stencil_spec.hpp>
#include <ozaki/utils/tensor.hpp>

#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

// Helper: given a concrete stencil type and M, constructs OzakiRunnerDirect<M, Dims...>.
template <int32_t M, typename CoeffT, size_t... Dims>
std::unique_ptr<OzakiRunnerBase>
make_runner_impl(const ozi::Stencil<CoeffT, Dims...> &stencil, const Tensor<Real> &input, int32_t iterations,
                 int32_t extra_msb, int32_t extra_lsb, const std::string &prefix, CoeffStorageType scaling_factor,
                 CoeffStorageType a_coeff, int32_t extra_msb_shift) {
  return std::make_unique<OzakiRunnerDirect<M, Dims...>>(stencil, input, iterations, extra_msb, extra_lsb, prefix,
                                                         scaling_factor, a_coeff, extra_msb_shift);
}

// Dispatches on M (1..12) for a given concrete stencil type.
template <typename CoeffT, size_t... Dims>
std::unique_ptr<OzakiRunnerBase>
dispatch_m(int m, const ozi::Stencil<CoeffT, Dims...> &stencil, const Tensor<Real> &input, int32_t iterations,
           int32_t extra_msb, int32_t extra_lsb, const std::string &prefix, CoeffStorageType scaling_factor,
           CoeffStorageType a_coeff, int32_t extra_msb_shift) {
  switch (m) {
  case 1:
    return make_runner_impl<1>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor, a_coeff,
                               extra_msb_shift);
  case 2:
    return make_runner_impl<2>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor, a_coeff,
                               extra_msb_shift);
  case 3:
    return make_runner_impl<3>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor, a_coeff,
                               extra_msb_shift);
  case 4:
    return make_runner_impl<4>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor, a_coeff,
                               extra_msb_shift);
  case 5:
    return make_runner_impl<5>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor, a_coeff,
                               extra_msb_shift);
  case 6:
    return make_runner_impl<6>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor, a_coeff,
                               extra_msb_shift);
  case 7:
    return make_runner_impl<7>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor, a_coeff,
                               extra_msb_shift);
  case 8:
    return make_runner_impl<8>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor, a_coeff,
                               extra_msb_shift);
  case 9:
    return make_runner_impl<9>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor, a_coeff,
                               extra_msb_shift);
  case 10:
    return make_runner_impl<10>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor, a_coeff,
                                extra_msb_shift);
  case 11:
    return make_runner_impl<11>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor, a_coeff,
                                extra_msb_shift);
  case 12:
    return make_runner_impl<12>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor, a_coeff,
                                extra_msb_shift);
  default:
    throw std::invalid_argument("  Stencil requires M=" + std::to_string(m) +
                                " slices, but only M <= 12 is supported. "
                                "Scaled integer coefficients are too large.");
  }
}

// Factory: creates the correctly-typed OzakiRunnerDirect from a runtime StencilSpec.
// Performs shape dispatch (std::visit) and M dispatch (switch) once at construction.
inline std::unique_ptr<OzakiRunnerBase> make_ozaki_runner(const StencilSpec &spec, const Tensor<Real> &input,
                                                          int32_t iterations, int32_t extra_msb, int32_t extra_lsb,
                                                          const std::string &prefix) {
  auto params = ozi::compute_stencil_params(spec);
  auto variant = ozi::make_stencil(spec);

  std::cout << std::format("  Using stencil {}: #Slices = {}, a = {}, extra_msb_shift = {}\n", spec.name, params.m,
                           params.a_coeff, params.extra_msb_shift);

  return std::visit(
      [&](auto &stencil) -> std::unique_ptr<OzakiRunnerBase> {
        return dispatch_m(params.m, stencil, input, iterations, extra_msb, extra_lsb, prefix, params.scaling_factor,
                          params.a_coeff, params.extra_msb_shift);
      },
      variant);
}

#endif // HEADER_RUNNER_FACTORY_CUH
