#ifndef HEADER_STENCIL_SPEC_HPP
#define HEADER_STENCIL_SPEC_HPP

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <quadmath.h>

// Coefficients are kept in binary128 so every supported file precision is exact.
using CoeffStorageType = __float128;

enum class OriginalPrecision { Int8, Float, Double, LongDouble, Float128 };

struct StencilSpec {
  std::string name;
  std::vector<int> shape;
  std::vector<int> offsets;
  std::vector<CoeffStorageType> coeffs;
  OriginalPrecision specified_precision = OriginalPrecision::Double;
};

// Reads a stencil specification from a text file.
// Format:
//   # comment
//   name HeatEquation
//   shape 3 3
//   offsets 1 1
//   precision double
//   coeffs
//   0.0  3.0  0.0
//   3.0  4.0  3.0
//   0.0  3.0  0.0
StencilSpec read_stencil_file(const std::string &path);

#endif // HEADER_STENCIL_SPEC_HPP
