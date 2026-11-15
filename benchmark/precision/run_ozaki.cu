#include <ozaki/gpu/gpu_checks.cuh>
#include <ozaki/gpu/stencil_integrator.cuh>
#include <ozaki/utils/stencil_spec.hpp>
#include <ozaki/utils/tensor.hpp>

#include <boost/program_options.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
  std::string output_dir;
  std::string output_prefix;
  std::string input_file;
  std::string stencil_file;
  int32_t slice_limit = 24;
  std::vector<size_t> timesteps;

  po::options_description options("run-ozaki: run OzakiStencil integration");
  options.add_options()("help,h", "Produce help message")("input,i", po::value<std::string>(&input_file)->required(),
                                                          "Path to input tensor file")(
      "dir", po::value<std::string>(&output_dir)->default_value("./data/tensors/"), "Output directory")(
      "prefix,p", po::value<std::string>(&output_prefix)->default_value("oz"),
      "Output file prefix")("stencil-file", po::value<std::string>(&stencil_file)->required(),
                            "Path to stencil text file")(
      "timesteps,T", po::value<std::vector<size_t>>(&timesteps)->multitoken(), "Requested timesteps")(
      "slice-limit,s", po::value<int32_t>(&slice_limit)->default_value(24), "Maximum data slices (2..31)");

  po::variables_map variables;
  try {
    po::store(po::parse_command_line(argc, argv, options), variables);
    if (variables.count("help") != 0U) {
      std::cout << options << '\n';
      return 0;
    }
    po::notify(variables);
    if (timesteps.empty()) {
      timesteps = {1000};
    }
    if (slice_limit < 2) {
      throw std::invalid_argument("--slice-limit must be at least two");
    }
  } catch (const std::exception &error) {
    std::cerr << "Command Line Error: " << error.what() << "\nUse --help to see usage.\n";
    return 1;
  }

  try {
    using Real = ozaki::gpu::IntegrationReal;
    const Tensor<Real> input = read_tensor_as<Real>(input_file);
    const StencilSpec spec = read_stencil_file(stencil_file);

    for (const size_t requested_timesteps : timesteps) {
      if (requested_timesteps > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::overflow_error("Requested timesteps exceed int32_t limits.");
      }
      std::cerr << "Running Ozaki for '" << spec.name << "' for " << requested_timesteps << " timesteps...\n";
      for (int32_t slices = 2; slices <= std::min(slice_limit, 31); ++slices) {
        ozaki::gpu::StencilIntegrator integrator(spec, input, slices);
        integrator.integrate(static_cast<int32_t>(requested_timesteps));
        std::vector<Real> result = integrator.reconstruct();
        const std::string path = output_dir + output_prefix + "_" + spec.name + "_" +
                                 std::to_string(requested_timesteps) + "_" + std::to_string(slices) + ".dat";
        write_tensor_cast<__float128>(path, input.shape, result);
        std::cerr << "  Completed: " << slices << " slices (saved at " << path << ")\n";
      }
    }
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
