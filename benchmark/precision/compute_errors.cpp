#include <ozaki/utils/error_metrics.hpp>
#include <ozaki/utils/tensor.hpp>

#include <format>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

int main(int argc, char *argv[]) {
  std::string reference_file;
  std::string result_file;

  po::options_description desc("compute-errors: compute RMSE between reference and result tensors");
  // clang-format off
  desc.add_options()
    ("help,h", "Produce help message")
    ("expected,e", po::value<std::string>(&reference_file)->required(),
     "Path to reference tensor file")
    ("actual,a", po::value<std::string>(&result_file)->required(),
     "Paths to result tensor file")
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
    std::cerr << "Command Line Error: " << e.what() << "\n";
    std::cerr << "Use -h or --help to see usage.\n";
    return 1;
  }

  try {
    // Load reference tensor at highest available precision
#ifdef __GNUC__
    auto ref = read_tensor<__float128>(reference_file);
    auto result = read_tensor<__float128>(result_file);
    __float128 error = calculate_rmse(ref.data, result.data);
#else
    auto ref = read_tensor<long double>(reference_file);
    auto result = read_tensor<long double>(result_files[i]);
    long double error = calculate_rmse(ref.data, result.data);
#endif

    std::cout << std::format("{}\n", error);

  } catch (const std::exception &e) {
    std::cerr << std::format("Error: {}\n", e.what());
    return 1;
  }

  return 0;
}
