#include <ozaki/utils/stencil_spec.hpp>

#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <quadmath.h>

namespace {

size_t checked_shape_product(const std::vector<int> &shape, const std::string &context) {
  if (shape.empty())
    throw std::invalid_argument(context + ": shape must not be empty");
  size_t product = 1;
  for (const int extent : shape) {
    if (extent <= 0)
      throw std::invalid_argument(context + ": shape extents must be positive");
    if (product > std::numeric_limits<size_t>::max() / static_cast<size_t>(extent))
      throw std::overflow_error(context + ": shape product overflows size_t");
    product *= static_cast<size_t>(extent);
  }
  return product;
}

} // namespace

StencilSpec read_stencil_file(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open())
    throw std::runtime_error("Cannot open stencil file: " + path);

  StencilSpec spec;
  spec.name = "unnamed";
  bool reading_coeffs = false;

  std::string line;
  while (std::getline(f, line)) {
    // Trim leading whitespace
    size_t start = line.find_first_not_of(" \t");
    if (start == std::string::npos)
      continue;
    line = line.substr(start);

    if (line.empty() || line[0] == '#')
      continue;

    if (reading_coeffs) {
      std::istringstream ss(line);
      std::string token;
      while (ss >> token) {
        spec.coeffs.push_back(strtoflt128(token.c_str(), nullptr));
      }
      continue;
    }

    std::istringstream ss(line);
    std::string key;
    ss >> key;

    if (key == "name") {
      ss >> spec.name;
    } else if (key == "shape") {
      int d;
      while (ss >> d)
        spec.shape.push_back(d);
    } else if (key == "offsets") {
      int o;
      while (ss >> o)
        spec.offsets.push_back(o);
    } else if (key == "precision") {
      std::string prec;
      ss >> prec;
      if (prec == "int8")
        spec.specified_precision = OriginalPrecision::Int8;
      else if (prec == "float")
        spec.specified_precision = OriginalPrecision::Float;
      else if (prec == "double")
        spec.specified_precision = OriginalPrecision::Double;
      else if (prec == "longdouble")
        spec.specified_precision = OriginalPrecision::LongDouble;
      else if (prec == "fp128")
        spec.specified_precision = OriginalPrecision::Float128;
      else
        throw std::invalid_argument("Unknown precision: " + prec);
    } else if (key == "coeffs") {
      reading_coeffs = true;
    }
  }

  // Default offsets if not specified
  if (spec.offsets.empty()) {
    for (int d : spec.shape)
      spec.offsets.push_back(d / 2);
  }

  // Validate
  const size_t expected_size = checked_shape_product(spec.shape, "Stencil file '" + path + "'");
  if (spec.coeffs.size() != expected_size)
    throw std::invalid_argument("Stencil file '" + path + "': coefficient count (" +
                                std::to_string(spec.coeffs.size()) + ") doesn't match shape product (" +
                                std::to_string(expected_size) + ")");

  return spec;
}
