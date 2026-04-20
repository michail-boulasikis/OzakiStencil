#include "runner_factory.cuh"

#include <ozaki/utils/stencil_registry.hpp>
#include <ozaki/utils/stencil_spec.hpp>
#include <ozaki/utils/tensor.hpp>

#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

int main(int argc, char *argv[]) {
  std::string output_dir;
  std::string output_prefix;
  std::string input_file;
  std::string stencil_arg;
  std::string stencil_file;
  std::string algorithm_str;
  int32_t slice_limit;
  std::vector<size_t> timesteps;
  int32_t extra_msb;
  int32_t extra_lsb;
  bool all_stencils = false;
  bool save_tensors = false;

  po::options_description desc("run-ozaki: run Ozaki stencil GPU integration");
  // clang-format off
  desc.add_options()
    ("help,h", "Produce help message")
    ("input,i", po::value<std::string>(&input_file)->required(),
     "Path to input tensor file")
    ("dir", po::value<std::string>(&output_dir)->default_value("./data/tensors/"),
     "Directory for output tensor files")
    ("prefix,p", po::value<std::string>(&output_prefix)->default_value("oz"),
     "Output file prefix")
    ("stencil", po::value<std::string>(&stencil_arg),
     "Inline stencil spec: \"RxC:c0,c1,...\"")
    ("stencil-file", po::value<std::string>(&stencil_file),
     "Path to stencil text file")
    ("all-stencils", "Use all default stencil configurations")
    ("timesteps,T", po::value<std::vector<size_t>>(&timesteps)->multitoken(),
     "Timesteps to compute (e.g., -T 100 1000 10000)")
    ("slice-limit,s", po::value<int32_t>(&slice_limit)->default_value(24),
     "Maximum slice count to test (starts from 2)")
    ("extra-msb", po::value<int32_t>(&extra_msb)->default_value(0),
     "Extra MSB headroom for dynamic range")
    ("extra-lsb", po::value<int32_t>(&extra_lsb)->default_value(0),
     "Extra LSB headroom for dynamic range")
    ("save-tensors,S", "Save every output tensor to disk")
    ("algorithm,A", po::value<std::string>(&algorithm_str)->default_value("direct"),
     "Algorithm: 'direct' (scalar stencil_v1) or 'convstencil' (WMMA tensor core, 2D 7x7 only)")
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
    save_tensors = vm.count("save-tensors") > 0;

    if (timesteps.empty())
      timesteps = {1000};

  } catch (const po::error &e) {
    std::cerr << "Command Line Error: " << e.what() << "\n";
    std::cerr << "Use -h or --help to see usage.\n";
    return 1;
  }

  Algorithm algo = Algorithm::Direct;
  if (algorithm_str == "convstencil")
    algo = Algorithm::ConvStencil;
  else if (algorithm_str != "direct")
    throw std::invalid_argument("Unknown algorithm: " + algorithm_str);

  try {
    NVTX_PUSH("Initialization");
    Tensor<__float128> input = read_tensor<__float128>(input_file);
    NVTX_POP();

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
      NVTX_PUSH(spec.name.c_str());

      for (size_t timestep : timesteps) {
        NVTX_PUSH((std::to_string(timestep) + " Timesteps").c_str());

        std::cerr << "Using algorithm " << algorithm_str << "\n";
        std::cerr << "Running Ozaki for '" << spec.name << "' for " << timestep << " timesteps...\n";

        auto runner =
            make_ozaki_runner(spec, input, static_cast<int32_t>(timestep), extra_msb, extra_lsb, output_dir, algo);

        for (int32_t s = 2; s <= slice_limit && s < runner->get_max_slices(); ++s) {
          std::string output_path = output_dir + output_prefix + "_" + spec.name + "_" + std::to_string(timestep) +
                                    "_" + std::to_string(s) + ".dat";
          runner->run_experiment(s, output_path, save_tensors);
          if (save_tensors)
            std::cerr << "  Completed: " << s << " Slices (saved at " << output_path << ")\n";
          else
            std::cerr << "  Completed: " << s << " Slices\n";
        }

        NVTX_POP();
      }

      NVTX_POP();
    }

    std::cerr << "Done.\n";

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
