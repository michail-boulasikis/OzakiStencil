#include <ozaki/core/device_memory.cuh>
#include <ozaki/gpu/gpu_checks.cuh>
#include <ozaki/gpu/naive_stencil.cuh>
#include <ozaki/gpu/stencil_integrator.cuh>
#include <ozaki/utils/stencil_registry.hpp>
#include <ozaki/utils/stencil_spec.hpp>
#include <ozaki/utils/tensor.hpp>
#include <ozaki/utils/zero_boundary.hpp>

#include <boost/program_options.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace po = boost::program_options;

namespace {

struct Config {
  std::vector<size_t> shape;
  int32_t timesteps = 100;
  int32_t warmup = 10;
  int32_t slices = 10;
};

std::vector<size_t> parse_shape(const std::string &text) {
  std::vector<size_t> shape;
  std::istringstream input(text);
  std::string extent;
  while (std::getline(input, extent, 'x')) {
    shape.push_back(std::stoull(extent));
  }
  if (shape.size() != 2U && shape.size() != 3U) {
    throw std::invalid_argument("Shape must be 2D or 3D.");
  }
  return shape;
}

size_t volume(const std::vector<size_t> &shape) {
  size_t result = 1;
  for (const size_t extent : shape) {
    if (extent == 0 || result > std::numeric_limits<size_t>::max() / extent) {
      throw std::overflow_error("Shape volume overflows size_t.");
    }
    result *= extent;
  }
  return result;
}

bool selected(const std::string &algorithms, std::string_view name) {
  if (algorithms == "all") {
    return true;
  }
  std::istringstream input(algorithms);
  std::string token;
  while (std::getline(input, token, ',')) {
    if (token == name || (token == "naive" && name.starts_with("naive-"))) {
      return true;
    }
  }
  return false;
}

class Events {
public:
  Events() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~Events() {
    (void)cudaEventDestroy(start_);
    (void)cudaEventDestroy(stop_);
  }
  float measure_start(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(start_, stream));
    return 0.0F;
  }
  float measure_stop(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(stop_, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds;
  }

private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

double throughput(size_t cells, int32_t timesteps, float milliseconds) {
  return milliseconds > 0.0F ? static_cast<double>(cells) * timesteps / (static_cast<double>(milliseconds) * 1.0e6)
                             : 0.0;
}

template <typename T, class RawStencil> auto rebind_stencil(const RawStencil &raw) {
  using Stencil = ozaki::RebindStencil_t<RawStencil, T>;
  Stencil result;
  result.offsets = raw.offsets;
  for (size_t coefficient = 0; coefficient < RawStencil::size; ++coefficient) {
    result.coeffs[coefficient] = static_cast<T>(raw.coeffs[coefficient]);
  }
  return result;
}

template <typename T, size_t... Dims>
void launch_naive(core::DeviceArray<T> &output, core::DeviceArray<T> &input, const ozaki::Stencil<T, Dims...> &,
                  const typename ozaki::Stencil<T, Dims...>::shape_type &shape) {
  constexpr size_t dimensions = sizeof...(Dims);
  const dim3 block = dimensions == 2 ? dim3(32, 8) : dim3(8, 8, 4);
  const dim3 grid(static_cast<unsigned>((shape[dimensions - 1] + block.x - 1) / block.x),
                  static_cast<unsigned>((shape[dimensions - 2] + block.y - 1) / block.y),
                  dimensions == 3 ? static_cast<unsigned>((shape[0] + block.z - 1) / block.z) : 1U);
  ozaki::naive_stencil<T, Dims...><<<grid, block>>>(output.get(), input.get(), shape);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T, class RawStencil>
double bench_naive(const RawStencil &raw, const Tensor<ozaki::gpu::IntegrationReal> &source, const Config &config) {
  auto stencil = rebind_stencil<T>(raw);
  using Stencil = decltype(stencil);
  typename Stencil::shape_type shape{};
  for (size_t dimension = 0; dimension < Stencil::dims; ++dimension) {
    shape[dimension] = source.shape[dimension];
  }
  Tensor<T> host;
  host.shape = source.shape;
  host.data.resize(source.data.size());
  std::transform(source.data.begin(), source.data.end(), host.data.begin(),
                 [](auto value) { return static_cast<T>(value); });
  host = zero_boundary_copy(host, Stencil::shape[0] / 2);

  core::DeviceArray<T> first{std::span<const T>(host.data)};
  core::DeviceArray<T> second(host.data.size());
  second.zero();
  CUDA_CHECK(cudaMemcpyToSymbol(ozaki::stencil_cmem, &stencil, sizeof(stencil)));
  for (int32_t step = 0; step < config.warmup; ++step) {
    launch_naive(second, first, stencil, shape);
    first.swap(second);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  first.copy_from_host(host.data);
  second.zero();

  Events events;
  events.measure_start(nullptr);
  for (int32_t step = 0; step < config.timesteps; ++step) {
    launch_naive(second, first, stencil, shape);
    first.swap(second);
  }
  const float elapsed = events.measure_stop(nullptr);
  return throughput(host.data.size(), config.timesteps, elapsed);
}

double bench_ozaki(const StencilSpec &spec, const Tensor<ozaki::gpu::IntegrationReal> &input, const Config &config) {
  ozaki::gpu::StencilIntegrator integrator(spec, input, config.slices);
  integrator.integrate(config.warmup);
  integrator.synchronize();
  integrator.reset();
  Events events;
  events.measure_start(integrator.stream());
  integrator.integrate(config.timesteps);
  const float elapsed = events.measure_stop(integrator.stream());
  return throughput(input.data.size(), config.timesteps, elapsed);
}

} // namespace

int main(int argc, char **argv) {
  po::options_description options("bench-stencil options");
  options.add_options()                                                                 //
      ("help,h", "Show help")                                                           //
      ("shape,S", po::value<std::string>()->required(), "Input grid shape, e.g. 1024x1024") //
      ("stencil-file", po::value<std::string>()->required(), "Stencil file")            //
      ("timesteps,T", po::value<int32_t>()->default_value(100), "Measured timesteps")   //
      ("warmup,W", po::value<int32_t>()->default_value(10), "Warmup timesteps")         //
      ("slices,s", po::value<int32_t>()->default_value(10), "Ozaki data slices")        //
      ("algorithm,A", po::value<std::string>()->default_value("all"),
       "Algorithms: ozaki, naive-fp32, naive-fp64, naive-fp128, naive, all");

  po::variables_map variables;
  try {
    po::store(po::parse_command_line(argc, argv, options), variables);
    if (variables.count("help") != 0U) {
      std::cout << options << '\n';
      return 0;
    }
    po::notify(variables);

    Config config;
    config.shape = parse_shape(variables["shape"].as<std::string>());
    config.timesteps = variables["timesteps"].as<int32_t>();
    config.warmup = variables["warmup"].as<int32_t>();
    config.slices = variables["slices"].as<int32_t>();
    if (config.timesteps <= 0 || config.warmup < 0 || config.slices < 1 || config.slices > 31) {
      throw std::invalid_argument("Invalid timing or slice-count arguments.");
    }

    const StencilSpec spec = read_stencil_file(variables["stencil-file"].as<std::string>());
    if (config.shape.size() != spec.shape.size()) {
      throw std::invalid_argument("Benchmark shape rank does not match stencil rank.");
    }

    using Real = ozaki::gpu::IntegrationReal;
    Tensor<Real> input;
    input.shape = config.shape;
    input.data.resize(volume(config.shape));
    std::mt19937_64 generator(42);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (Real &value : input.data) {
      value = static_cast<Real>(distribution(generator));
    }

    const std::string algorithms = variables["algorithm"].as<std::string>();
    const auto typed = ozaki::make_stencil(spec);
    if (selected(algorithms, "naive-fp32")) {
      const double value =
          std::visit([&](const auto &stencil) { return bench_naive<float>(stencil, input, config); }, typed);
      std::cout << "Naive FP32 GStPt/s " << value << '\n';
    }
    if (selected(algorithms, "naive-fp64")) {
      const double value =
          std::visit([&](const auto &stencil) { return bench_naive<double>(stencil, input, config); }, typed);
      std::cout << "Naive FP64 GStPt/s " << value << '\n';
    }
#if defined(OZAKI_CUDA_ARCH) && OZAKI_CUDA_ARCH >= 100
    if (selected(algorithms, "naive-fp128")) {
      const double value =
          std::visit([&](const auto &stencil) { return bench_naive<__float128>(stencil, input, config); }, typed);
      std::cout << "Naive FP128 GStPt/s " << value << '\n';
    }
#endif
    if (selected(algorithms, "ozaki")) {
      std::cout << "OzakiStencil GStPt/s " << bench_ozaki(spec, input, config) << '\n';
    }
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
