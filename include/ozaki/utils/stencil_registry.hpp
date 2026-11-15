#ifndef HEADER_STENCIL_REGISTRY_HPP
#define HEADER_STENCIL_REGISTRY_HPP

#include "../core/stencil.hpp"
#include "stencil_spec.hpp"

#include <cstddef>
#include <variant>

namespace ozaki {

// Variant over all supported stencil shapes.
// Adding a new shape requires one line here and a recompile.
using StencilVariant =
    std::variant<Stencil<CoeffStorageType, 3, 3>, Stencil<CoeffStorageType, 7, 7>, Stencil<CoeffStorageType, 15, 15>,
                 Stencil<CoeffStorageType, 3, 3, 3>, Stencil<CoeffStorageType, 5, 5, 5>>;

// Creates a typed Stencil from a runtime StencilSpec.
// Throws std::invalid_argument if the shape is not in the registry.
StencilVariant make_stencil(const StencilSpec &spec);

} // namespace ozaki

#endif // HEADER_STENCIL_REGISTRY_HPP
