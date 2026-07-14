#ifndef HEADER_STENCIL_SPEC_HPP
#define HEADER_STENCIL_SPEC_HPP

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef OZAKI_HAVE_FLOAT128
#if (defined(__GNUC__) && !defined(__NVCOMPILER) && !defined(__CUDA_ARCH__)) || __CUDA_ARCH__ >= 1000
#define OZAKI_HAVE_FLOAT128 1
#endif
#endif

#ifdef OZAKI_HAVE_FLOAT128
#include <quadmath.h>
using CoeffStorageType = __float128;
#else
using CoeffStorageType = long double;
#endif

enum class OriginalPrecision { Int8, Float, Double, LongDouble, Float128 };

struct StencilSpec {
  std::string name;
  std::vector<int> shape;
  std::vector<int> offsets;
  std::vector<CoeffStorageType> coeffs;
  OriginalPrecision specified_precision = OriginalPrecision::Double;
};

// Parses an inline stencil specification of the form "RxC:c0,c1,c2,..."
// Example: "3x3:0,3,0,3,4,3,0,3,0"
StencilSpec parse_stencil_arg(const std::string &arg);

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
