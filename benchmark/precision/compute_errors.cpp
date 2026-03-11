#include <ozaki/utils/error_metrics.hpp>
#include <ozaki/utils/tensor.hpp>

#include <iostream>
#include <string>
#include <algorithm>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#ifdef OZAKI_HAVE_FLOAT128
#define HIGH_PREC __float128
#define HIGH_PREC_MAX(X, Y) fmaxq(X, Y)
#define HIGH_PREC_ABS(X) fabsq(X)
#else
#define HIGH_PREC (long double)
#define HIGH_PREC_ABS(X) std::abs(X)
#define HIGH_PREC_MAX(X, Y) std::max(X, Y)
#endif

HIGH_PREC absmax(const Tensor<HIGH_PREC> &a) {
  HIGH_PREC amax = static_cast<HIGH_PREC>(-1.0);
  for (const auto &d : a.data)
    if (HIGH_PREC_ABS(d) > amax) {
      amax = d;
    }
  return amax;
}

void div_tensors(Tensor<HIGH_PREC> &a, HIGH_PREC amax) {
  std::for_each(a.data.begin(), a.data.end(), [&amax](HIGH_PREC &i) { i = i / amax; });
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
    // Load reference tensor at highest available precision
    auto ref = read_tensor<HIGH_PREC>(reference_file);
    auto result = read_tensor<HIGH_PREC>(result_file);
    if (scaled) {
      auto scale = absmax(ref);
      div_tensors(ref, scale);
      div_tensors(result, scale);
    }

    HIGH_PREC error = calculate_rmse(ref.data, result.data);
    std::cout << static_cast<long double>(error) << "\n";

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
