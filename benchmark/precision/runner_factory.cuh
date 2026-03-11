#ifndef HEADER_RUNNER_FACTORY_CUH
#define HEADER_RUNNER_FACTORY_CUH

#include "ozaki_runner.cuh"

#include <ozaki/utils/stencil_registry.hpp>
#include <ozaki/utils/stencil_spec.hpp>
#include <ozaki/utils/tensor.hpp>

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>

enum class Algorithm { Direct, ConvStencil };

// ==================== Direct (stencil_v1) dispatch ====================

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
  case 8:
    return make_runner_impl<8>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor, a_coeff,
                               extra_msb_shift);
  case 9:
    return make_runner_impl<9>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor, a_coeff,
                               extra_msb_shift);
  case 10:
    return make_runner_impl<10>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor, a_coeff,
                                extra_msb_shift);
  default:
    throw std::invalid_argument("  Stencil requires M=" + std::to_string(m) +
                                " slices, but M=" + std::to_string(m) +
                                " is not a supported instantiation.");
  }
}

// ==================== ConvStencil (WMMA) dispatch ====================

// Helper: constructs OzakiRunnerConvStencil<M> only for 2D 7x7 stencils.
// Uses if constexpr to avoid instantiation errors for other shapes.
template <int32_t M, typename CoeffT, size_t... Dims>
std::unique_ptr<OzakiRunnerBase>
make_convstencil_runner_impl(const ozi::Stencil<CoeffT, Dims...> &stencil, const Tensor<Real> &input,
                             int32_t iterations, int32_t extra_msb, int32_t extra_lsb, const std::string &prefix,
                             CoeffStorageType scaling_factor, CoeffStorageType a_coeff, int32_t extra_msb_shift) {
  if constexpr (sizeof...(Dims) == 2) {
    constexpr auto shape = ozi::Stencil<CoeffT, Dims...>::shape;
    if constexpr (shape[0] == 7 && shape[1] == 7) {
      return std::make_unique<OzakiRunnerConvStencil<M>>(stencil, input, iterations, extra_msb, extra_lsb, prefix,
                                                         scaling_factor, a_coeff, extra_msb_shift);
    }
  }
  throw std::invalid_argument("ConvStencil algorithm only supports 2D 7x7 stencils");
}

// Dispatches on M (1..12) for ConvStencil.
template <typename CoeffT, size_t... Dims>
std::unique_ptr<OzakiRunnerBase>
dispatch_m_convstencil(int m, const ozi::Stencil<CoeffT, Dims...> &stencil, const Tensor<Real> &input,
                       int32_t iterations, int32_t extra_msb, int32_t extra_lsb, const std::string &prefix,
                       CoeffStorageType scaling_factor, CoeffStorageType a_coeff, int32_t extra_msb_shift) {
  switch (m) {
  case 1:
    return make_convstencil_runner_impl<1>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor,
                                           a_coeff, extra_msb_shift);
  case 2:
    return make_convstencil_runner_impl<2>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor,
                                           a_coeff, extra_msb_shift);
  case 3:
    return make_convstencil_runner_impl<3>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor,
                                           a_coeff, extra_msb_shift);
  case 4:
    return make_convstencil_runner_impl<4>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor,
                                           a_coeff, extra_msb_shift);
  case 8:
    return make_convstencil_runner_impl<8>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor,
                                           a_coeff, extra_msb_shift);
  case 9:
    return make_convstencil_runner_impl<9>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor,
                                           a_coeff, extra_msb_shift);
  case 10:
    return make_convstencil_runner_impl<10>(stencil, input, iterations, extra_msb, extra_lsb, prefix, scaling_factor,
                                            a_coeff, extra_msb_shift);
  default:
    throw std::invalid_argument("  Stencil requires M=" + std::to_string(m) +
                                " slices, but M=" + std::to_string(m) +
                                " is not a supported instantiation.");
  }
}

// ==================== Factory ====================

// Factory: creates the correctly-typed runner from a runtime StencilSpec.
// Performs shape dispatch (std::visit) and M dispatch (switch) once at construction.
inline std::unique_ptr<OzakiRunnerBase> make_ozaki_runner(const StencilSpec &spec, const Tensor<Real> &input,
                                                          int32_t iterations, int32_t extra_msb, int32_t extra_lsb,
                                                          const std::string &prefix,
                                                          Algorithm algo = Algorithm::Direct) {
  auto params = ozi::compute_stencil_params(spec);
  auto variant = ozi::make_stencil(spec);

  // Validate that stencil dimensionality matches input tensor dimensionality
  std::visit(
      [&](const auto &s) {
        constexpr size_t stencil_ndims = std::decay_t<decltype(s)>::dims;
        if (stencil_ndims != input.shape.size())
          throw std::invalid_argument("Stencil '" + spec.name + "' has " + std::to_string(stencil_ndims) +
                                      " dimensions but input tensor has " + std::to_string(input.shape.size()) +
                                      " dimensions");
      },
      variant);

  {
    std::ostringstream oss;
    oss << "  Using stencil " << spec.name << ": #Slices = " << params.m
        << ", a = " << static_cast<long double>(params.a_coeff) << ", extra_msb_shift = " << params.extra_msb_shift << "\n";
    std::cout << oss.str();
  }

  return std::visit(
      [&](auto &stencil) -> std::unique_ptr<OzakiRunnerBase> {
        if (algo == Algorithm::ConvStencil) {
          return dispatch_m_convstencil(params.m, stencil, input, iterations, extra_msb, extra_lsb, prefix,
                                        params.scaling_factor, params.a_coeff, params.extra_msb_shift);
        }
        return dispatch_m(params.m, stencil, input, iterations, extra_msb, extra_lsb, prefix, params.scaling_factor,
                          params.a_coeff, params.extra_msb_shift);
      },
      variant);
}

#endif // HEADER_RUNNER_FACTORY_CUH
