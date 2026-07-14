#include <ozaki/utils/stencil_spec.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>

StencilSpec parse_stencil_arg(const std::string &arg) {
  StencilSpec spec;

  auto colon = arg.find(':');
  if (colon == std::string::npos)
    throw std::invalid_argument("Stencil arg must be 'RxC:c0,c1,...', got: " + arg);

  // Parse shape from "RxC" or "RxCxD"
  std::string shape_str = arg.substr(0, colon);
  std::istringstream shape_ss(shape_str);
  std::string dim_str;
  while (std::getline(shape_ss, dim_str, 'x')) {
    spec.shape.push_back(std::stoi(dim_str));
  }

  // Default offsets: center of each dimension
  for (int d : spec.shape) {
    spec.offsets.push_back(d / 2);
  }

  // Parse coefficients
  std::string coeffs_str = arg.substr(colon + 1);
  std::istringstream coeffs_ss(coeffs_str);
  std::string coeff_str;
  while (std::getline(coeffs_ss, coeff_str, ',')) {
#ifdef OZAKI_HAVE_FLOAT128
    spec.coeffs.push_back(strtoflt128(coeff_str.c_str(), nullptr));
#else
    spec.coeffs.push_back(static_cast<CoeffStorageType>(std::stold(coeff_str)));
#endif
  }

  // Validate size
  size_t expected_size = 1;
  for (int d : spec.shape)
    expected_size *= static_cast<size_t>(d);
  if (spec.coeffs.size() != expected_size)
    throw std::invalid_argument("Coefficient count (" + std::to_string(spec.coeffs.size()) +
                                ") doesn't match shape product (" + std::to_string(expected_size) + ")");

  // Infer precision from values: if all round-trip through int8, mark as Int8
  bool all_int8 = true;
  for (const auto &c : spec.coeffs) {
    auto rounded = static_cast<CoeffStorageType>(static_cast<int8_t>(static_cast<long double>(c)));
    if (rounded != c) {
      all_int8 = false;
      break;
    }
  }
  spec.specified_precision = all_int8 ? OriginalPrecision::Int8 : OriginalPrecision::Double;
  spec.name = "inline";

  return spec;
}

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
#ifdef OZAKI_HAVE_FLOAT128
        spec.coeffs.push_back(strtoflt128(token.c_str(), nullptr));
#else
        spec.coeffs.push_back(static_cast<CoeffStorageType>(std::stold(token)));
#endif
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
  size_t expected_size = 1;
  for (int d : spec.shape)
    expected_size *= static_cast<size_t>(d);
  if (spec.coeffs.size() != expected_size)
    throw std::invalid_argument("Stencil file '" + path + "': coefficient count (" +
                                std::to_string(spec.coeffs.size()) + ") doesn't match shape product (" +
                                std::to_string(expected_size) + ")");

  return spec;
}

