#include <ozaki/utils/stencil_registry.hpp>

#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

namespace ozaki {

StencilVariant make_stencil(const StencilSpec &spec) {
  if (spec.offsets.size() != spec.shape.size()) {
    throw std::invalid_argument("Stencil offset rank does not match its shape.");
  }
  size_t coefficient_count = 1;
  for (const int extent : spec.shape) {
    if (extent <= 0 || coefficient_count > std::numeric_limits<size_t>::max() / static_cast<size_t>(extent)) {
      throw std::invalid_argument("Stencil shape contains an invalid or overflowing extent.");
    }
    coefficient_count *= static_cast<size_t>(extent);
  }
  if (spec.shape.empty() || spec.coeffs.size() != coefficient_count) {
    throw std::invalid_argument("Stencil coefficient count does not match its shape.");
  }

  auto try_make = [&]<typename T, size_t... Dims>(Stencil<T, Dims...> *) -> bool {
    constexpr std::array<size_t, sizeof...(Dims)> required_shape = {Dims...};
    if (spec.shape.size() != sizeof...(Dims))
      return false;
    for (size_t i = 0; i < sizeof...(Dims); ++i) {
      if (static_cast<size_t>(spec.shape[i]) != required_shape[i])
        return false;
    }
    return true;
  };

  auto populate = [&]<typename T, size_t... Dims>(Stencil<T, Dims...> &s) {
    for (size_t i = 0; i < sizeof...(Dims); ++i)
      s.offsets[i] = static_cast<size_t>(spec.offsets[i]);
    for (size_t i = 0; i < s.size; ++i)
      s.coeffs[i] = spec.coeffs[i];
  };

  StencilVariant result;
  bool found = false;

  auto try_variant = [&]<typename Alt>(Alt *tag) {
    if (!found && try_make(tag)) {
      Alt s;
      populate(s);
      result = s;
      found = true;
    }
  };

  std::apply([&](auto *...tags) { (try_variant(tags), ...); },
             std::tuple{static_cast<Stencil<CoeffStorageType, 3, 3> *>(nullptr),
                        static_cast<Stencil<CoeffStorageType, 7, 7> *>(nullptr),
                        static_cast<Stencil<CoeffStorageType, 15, 15> *>(nullptr),
                        static_cast<Stencil<CoeffStorageType, 3, 3, 3> *>(nullptr),
                        static_cast<Stencil<CoeffStorageType, 5, 5, 5> *>(nullptr)});

  if (!found) {
    std::string shape_str;
    for (size_t i = 0; i < spec.shape.size(); ++i) {
      if (i > 0)
        shape_str += "x";
      shape_str += std::to_string(spec.shape[i]);
    }
    throw std::invalid_argument("Unsupported stencil shape: " + shape_str +
                                ". Supported shapes: 3x3, 7x7, 15x15, 3x3x3, 5x5x5.");
  }

  return result;
}
} // namespace ozaki
