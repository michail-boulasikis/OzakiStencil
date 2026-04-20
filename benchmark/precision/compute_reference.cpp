#include "reference_solver.hpp"

#include <ozaki/utils/error_metrics.hpp>
#include <ozaki/utils/stencil_registry.hpp>
#include <ozaki/utils/stencil_spec.hpp>
#include <ozaki/utils/tensor.hpp>

#include <iostream>
#include <string>
#include <variant>
#include <vector>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

int main(int argc, char *argv[]) {
  std::string output_dir;
  std::string output_prefix;
  std::string input_file;
  std::string stencil_arg;
  std::string stencil_file;
  std::vector<size_t> timesteps;
  bool all_stencils = false;
  bool compute_float_ref = false;
  bool compute_double_ref = false;
  bool compute_longdouble_ref = false;
  bool compute_fp128_ref = false;

  po::options_description desc("compute-reference: produce stable reference tensor files");
  // clang-format off
  desc.add_options()
    ("help,h", "Produce help message")
    ("input,i", po::value<std::string>(&input_file)->required(),
     "Path to input tensor file")
    ("dir", po::value<std::string>(&output_dir)->default_value("./data/tensors/"),
     "Directory for output tensor files")
    ("prefix,p", po::value<std::string>(&output_prefix)->default_value("ref"),
     "Output file prefix")
    ("stencil", po::value<std::string>(&stencil_arg),
     "Inline stencil spec: \"RxC:c0,c1,...\"")
    ("stencil-file", po::value<std::string>(&stencil_file),
     "Path to stencil text file")
    ("all-stencils", "Use all default stencil configurations")
    ("timesteps,T", po::value<std::vector<size_t>>(&timesteps)->multitoken(),
     "Timesteps to compute (e.g., -T 100 1000 10000)")
    ("float", "Compute float-precision reference")
    ("double", "Compute double-precision reference")
    ("longdouble", "Compute long-double-precision reference")
    ("fp128", "Compute __float128-precision reference (default on GCC)")
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

    all_stencils = vm.count("all-stencils") > 0;
    compute_float_ref = vm.count("float") > 0;
    compute_double_ref = vm.count("double") > 0;
    compute_longdouble_ref = vm.count("longdouble") > 0;
    compute_fp128_ref = vm.count("fp128") > 0;

    // Default: fp128 if nothing specified
    if (!compute_float_ref && !compute_double_ref && !compute_longdouble_ref && !compute_fp128_ref)
      compute_fp128_ref = true;

    if (timesteps.empty())
      timesteps = {1000};

  } catch (const po::error &e) {
    std::cerr << "Command Line Error: " << e.what() << "\n";
    std::cerr << "Use -h or --help to see usage.\n";
    return 1;
  }

  try {
    Tensor<__float128> input = read_tensor<__float128>(input_file);

    // Collect stencil specifications
    std::vector<StencilSpec> specs;
    if (all_stencils) {
      specs = ozi::default_stencils();
    } else if (!stencil_arg.empty()) {
      specs.push_back(parse_stencil_arg(stencil_arg));
    } else if (!stencil_file.empty()) {
      specs.push_back(read_stencil_file(stencil_file));
    } else {
      specs = ozi::default_stencils();
    }

    for (const auto &spec : specs) {
      auto variant = ozi::make_stencil(spec);

      for (size_t timestep : timesteps) {
        std::cerr << "Computing references for '" << spec.name << "' at " << timestep << " timesteps...\n";

        auto compute_and_save = [&]<typename ComputeType>(const std::string &type_label) {
          auto result = std::visit(
              [&](const auto &stencil) {
                return ReferenceSolver::compute<ComputeType>(input, stencil, static_cast<int32_t>(timestep));
              },
              variant);

          std::string path =
              output_dir + output_prefix + "_" + spec.name + "_" + type_label + "_" + std::to_string(timestep) + ".dat";
          std::vector<__float128> q_result(result.size());
          for (size_t i = 0; i < result.size(); ++i)
            q_result[i] = static_cast<__float128>(result[i]);
          write_tensor(path, Tensor<__float128>{input.shape, q_result});
          std::cerr << "  Wrote: " << path << "\n";
        };

#ifdef OZAKI_HAVE_FLOAT128
        if (compute_fp128_ref)
          compute_and_save.template operator()<__float128>("fp128");
#endif
        if (compute_longdouble_ref)
          compute_and_save.template operator()<long double>("longdouble");
        if (compute_double_ref)
          compute_and_save.template operator()<double>("double");
        if (compute_float_ref)
          compute_and_save.template operator()<float>("float");
      }
    }

    std::cerr << "Done.\n";

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
