#include <ozaki/utils/tensor.hpp>
#include <ozaki/utils/zero_boundary.hpp>

#include <boost/program_options.hpp>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace po = boost::program_options;

// Generate an fp128 value by sampling a double and adding sub-double noise.
static __float128 rand_fp128(std::mt19937_64 &rng, std::normal_distribution<double> &gauss,
                             std::uniform_real_distribution<double> &unit) {
  double hi = gauss(rng);
  __float128 q = (__float128)hi;
  q += (__float128)hi * (__float128)(unit(rng) * 0x1p-53);
  q += (__float128)hi * (__float128)(unit(rng) * 0x1p-106);
  return q;
}

// Generates a Gaussian tensor whose halo of width `halo` is zero, matching the
// homogeneous Dirichlet boundary used by every solver in this project.
static Tensor<__float128> generate(std::vector<size_t> shape, size_t halo, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> gauss(0.0, 1.0);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);

  size_t total = 1;
  for (size_t s : shape)
    total *= s;

  std::vector<__float128> output(total);
  for (auto &v : output)
    v = rand_fp128(rng, gauss, unit);

  return zero_boundary_copy(Tensor<__float128>{shape, std::move(output)}, halo);
}

int main(int argc, char *argv[]) {
  std::string output_file;
  size_t n = 64;
  int ndim = 2;
  uint64_t seed = 42;
  // 7 is the halo of the widest supported stencil (15x15), so the default
  // output is a valid zero-boundary input for every stencil shape.
  size_t halo = 7;

  po::options_description desc("gen-tensor: generate FP128 Gaussian random tensor files");
  // clang-format off
    desc.add_options()
        ("help,h",   "Produce help message")
        ("output,o", po::value<std::string>(&output_file)->required(),
                     "Path to output binary tensor file")
        ("N",        po::value<size_t>(&n)->default_value(64),
                     "Grid size per dimension")
        ("dims,d",   po::value<int>(&ndim)->default_value(2),
                     "Dimensionality: 2 or 3")
        ("seed",     po::value<uint64_t>(&seed)->default_value(42),
                     "RNG seed")
        ("halo",     po::value<size_t>(&halo)->default_value(7),
                     "Width of the zero boundary (7 covers every supported stencil)")
        ;
  // clang-format on

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 0;
    }
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << "Error: " << e.what() << "\n";
    std::cerr << "Use -h or --help to see usage.\n";
    return 1;
  }

  if (ndim < 2 || ndim > 3) {
    std::cerr << "Error: --dims must be 2 or 3.\n";
    return 1;
  }

  try {
    std::vector<size_t> shape(ndim, n);
    Tensor<__float128> tensor = generate(shape, halo, seed);

    std::cerr << "Generated " << ndim << "D FP128 tensor, shape=[" << n;
    for (int d = 1; d < ndim; ++d)
      std::cerr << "x" << n;
    std::cerr << "], zero halo width " << halo << "\n";

    write_tensor(output_file, tensor);
    std::cerr << "Written to " << output_file << "\n";

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
