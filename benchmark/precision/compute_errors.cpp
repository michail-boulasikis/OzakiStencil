#include <ozaki/utils/error_metrics.hpp>
#include <ozaki/utils/tensor.hpp>

#include <algorithm>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

__float128 absmax(const Tensor<__float128> &a) {
  __float128 amax = static_cast<__float128>(0);
  for (const auto &d : a.data)
    if (fabsq(d) > amax) {
      amax = fabsq(d);
    }
  return amax;
}

void div_tensors(Tensor<__float128> &a, __float128 amax) {
  std::for_each(a.data.begin(), a.data.end(), [&amax](__float128 &i) { i = i / amax; });
}

int main(int argc, char *argv[]) {
  std::string reference_file;
  std::string result_file;
  bool scaled = false;

  po::options_description desc("compute-errors: compute RMSE between reference and result tensors");
  // clang-format off
  desc.add_options()
    ("help,h", "Produce help message")
    ("expected,e", po::value<std::string>(&reference_file)->required(),
     "Path to reference tensor file")
    ("actual,a", po::value<std::string>(&result_file)->required(),
     "Paths to result tensor file")
    ("scaled", "Scale tensors to [-1,1] before error computation")
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

    scaled = vm.count("scaled") > 0;

  } catch (const po::error &e) {
    std::cerr << "Command Line Error: " << e.what() << "\n";
    std::cerr << "Use -h or --help to see usage.\n";
    return 1;
  }

  try {
    // Load tensors strictly as __float128; both reference and Ozaki outputs are written in __float128.
    auto ref = read_tensor<__float128>(reference_file);
    auto result = read_tensor<__float128>(result_file);
    if (scaled) {
      auto scale = absmax(ref);
      // A zero reference tensor has no meaningful normalization scale. In
      // particular, stable zero-boundary shifts can become exactly zero after
      // enough iterations. Treat that case as an exact zero-error result
      // instead of producing NaNs from 0 / 0.
      if (scale == static_cast<__float128>(0.0)) {
        std::cout << "0,0\n";
        return 0;
      }
      div_tensors(ref, scale);
      div_tensors(result, scale);
    }

    __float128 error = calculate_rmse(ref.data, result.data);
    __float128 max_abs_diff = calculate_max_abs_diff(ref.data, result.data);
    std::cout << static_cast<long double>(error) << "," << static_cast<long double>(max_abs_diff) << "\n";

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
