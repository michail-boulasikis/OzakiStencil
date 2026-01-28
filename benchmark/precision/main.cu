#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <ozaki/core/stencil.hpp>
#include <ozaki/cpu/ozi.hpp>
#include <ozaki/gpu/ozi.cuh>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using Real = float;
using AccumulatorType = float;
using RealStencil = ozi::Stencil<Real, 3, 3>;
using IntStencil = ozi::Stencil<int8_t, 3, 3>;

struct Config {
#ifndef RDC_THREADS_PER_BLOCK
  static constexpr int RDC_THREADS = 512;
#else
  static constexpr int RDC_THREADS = RDC_THREADS_PER_BLOCK;
#endif
  static constexpr int STENCIL_ITERS_DEFAULT = 4096;
  static constexpr size_t STENCIL_BLOCK_DIM_X = 16;
  static constexpr size_t STENCIL_BLOCK_DIM_Y = 16;
  static constexpr std::array<int8_t, 9> DEFAULT_STENCIL_OZ = {0, 24, 0, 24, 32, 24, 0, 24, 0};
};

int32_t *d_flush_buffer = nullptr;
size_t flush_buffer_size = 0;

void invalidate_caches() {
  NVTX_PUSH("Flushing L2")
  if (d_flush_buffer == nullptr) {
    int device_id;
    int l2_cache_size;
    CUDA_CHECK(cudaGetDevice(&device_id));
    CUDA_CHECK(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, device_id));
    flush_buffer_size = l2_cache_size * 2;
    CUDA_CHECK(cudaMalloc(&d_flush_buffer, flush_buffer_size));
  }
  CUDA_CHECK(cudaMemset(d_flush_buffer, 0, flush_buffer_size));
  CUDA_CHECK(cudaDeviceSynchronize());
  NVTX_POP();
}

template <typename T> struct Tensor {
  std::vector<size_t> shape;
  std::vector<T> data;

  size_t size() const { return data.size(); }
  size_t width() const { return shape.empty() ? 0 : shape[0]; }
  size_t height() const { return shape.size() > 1 ? shape[1] : 1; }
};

template <typename T_Test, typename T_Ref>
long double calculate_rmse(const std::vector<T_Test> &test, const std::vector<T_Ref> &ref) {
  if (test.size() != ref.size()) {
    throw std::runtime_error("Size mismatch in MSE calculation");
  }
  long double err = 0.0;
  for (size_t i = 0; i < test.size(); ++i) {
    long double diff = static_cast<long double>(test[i]) - static_cast<long double>(ref[i]);
    err += diff * diff;
  }
  return std::sqrt(err / static_cast<long double>(test.size()));
}

template <typename T> Tensor<T> read_tensor(const std::string &filepath) {
  std::ifstream f(filepath, std::ios::in | std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error(std::format("Could not open file at {}.", filepath));
  }

  size_t n_dims;
  f.read(reinterpret_cast<char *>(&n_dims), sizeof(size_t));

  std::vector<size_t> shape(n_dims);
  f.read(reinterpret_cast<char *>(shape.data()), n_dims * sizeof(size_t));

  size_t n_elems = 1;
  for (const auto &dim : shape)
    n_elems *= dim;

  std::vector<T> data(n_elems);
  f.read(reinterpret_cast<char *>(data.data()), n_elems * sizeof(T));

  return {shape, data};
}

template <typename T> void write_tensor(const std::string &filepath, const Tensor<T> &tensor) {
  std::ofstream f(filepath, std::ios::out | std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error(std::format("Could not open file at {}.", filepath));
  }

  size_t n_dims = tensor.shape.size();
  f.write(reinterpret_cast<const char *>(&n_dims), sizeof(size_t));
  f.write(reinterpret_cast<const char *>(tensor.shape.data()), n_dims * sizeof(size_t));
  f.write(reinterpret_cast<const char *>(tensor.data.data()), tensor.size() * sizeof(T));
}

std::tuple<RealStencil, IntStencil, Real> create_stencils(const std::vector<int> &args) {
  std::array<int8_t, 9> coeffs_oz = Config::DEFAULT_STENCIL_OZ;
  if (!args.empty()) {
    if (args.size() == 9) {
      for (size_t i = 0; i < 9; ++i) {
        coeffs_oz[i] = static_cast<int8_t>(args[i]);
      }
    } else {
      std::cerr << std::format("Warning: Expected 9 stencil coefficients, but got {}. Using defaults.\n", args.size());
    }
  }
  const IntStencil heat_ozaki{coeffs_oz, {1, 1}};
  double l1 = heat_ozaki.template l1_norm<double>();
  Real scaling_factor = (l1 > 1e-9) ? (static_cast<Real>(1.0) / static_cast<Real>(l1)) : static_cast<Real>(1.0);

  std::array<Real, 9> coeffs;
  for (size_t i = 0; i < 9; ++i) {
    coeffs[i] = static_cast<Real>(coeffs_oz[i]) * scaling_factor;
  }

  const RealStencil heat{coeffs, {1, 1}};
  return {heat, heat_ozaki, scaling_factor};
}

struct RangeAnalyzer {
  static void analyze(const Tensor<Real> &input, const RealStencil &stencil, const int32_t iterations,
                      std::ofstream &outf) {
    NVTX_PUSH("Range Analysis");
    outf << "Time,Max_Exp,Min_Exp\n";

    ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>> *d_desc = nullptr;
    ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>> h_desc; // Host copy for reading back results
    Real *d_data = nullptr;
    size_t n_elems = input.size();

    CUDA_CHECK(cudaMalloc(&d_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>)));
    CUDA_CHECK(cudaMalloc(&d_data, n_elems * sizeof(Real)));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int32_t nThreads = Config::RDC_THREADS;
    const int32_t nBlocks = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / Config::RDC_THREADS;

    std::vector<Real> u_curr = input.data;
    std::vector<Real> u_next = u_curr;

    for (int32_t t = 0; t <= iterations; ++t) {
      CUDA_CHECK(cudaMemcpy(d_data, u_curr.data(), n_elems * sizeof(Real), cudaMemcpyHostToDevice));

      h_desc = ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>();
      CUDA_CHECK(cudaMemcpy(d_desc, &h_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>),
                            cudaMemcpyHostToDevice));

      ozigpu::dyn_range<<<nBlocks, nThreads>>>(d_desc, d_data, n_elems);
      CUDA_CHECK(cudaGetLastError());

      CUDA_CHECK(cudaMemcpy(&h_desc, d_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>),
                            cudaMemcpyDeviceToHost));

      outf << std::format("{},{},{}\n", t, h_desc.e_max, h_desc.e_min);
      if (t < iterations) {
        stencil.template apply_to<Real>(u_curr, u_next, {input.width(), input.height()});
        std::swap(u_curr, u_next);
      }
    }

    cudaFree(d_desc);
    cudaFree(d_data);
    NVTX_POP();
  }
};

struct ReferenceSolver {
  template <typename ComputeType>
  static std::vector<ComputeType> compute(const Tensor<Real> &input, const ozi::Stencil<Real, 3, 3> &stencil_def,
                                          const std::string &label = "Reference",
                                          const int32_t iterations = Config::STENCIL_ITERS_DEFAULT) {
    NVTX_PUSH(std::format("{} Solver", label).c_str());

    std::vector<ComputeType> u_curr(input.data.size());
    std::transform(input.data.begin(), input.data.end(), u_curr.begin(),
                   [](Real x) { return static_cast<ComputeType>(x); });

    std::vector<ComputeType> u_next = u_curr;

    ozi::Stencil<ComputeType, 3, 3> stencil_typed;
    stencil_typed.offsets = stencil_def.offsets;
    for (size_t i = 0; i < stencil_def.coeffs.size(); ++i) {
      stencil_typed.coeffs[i] = static_cast<ComputeType>(stencil_def.coeffs[i]);
    }

    for (size_t i = 0; i < iterations + 1; ++i) {
      stencil_typed.template apply_to<ComputeType>(u_curr, u_next, {input.width(), input.height()});
      std::swap(u_curr, u_next);
    }

    NVTX_POP();
    return u_curr;
  }
};

class OzakiRunner {
  int8_t *d_slices_1 = nullptr, *d_slices_2 = nullptr;

  ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>> *d_desc = nullptr;
  ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>> h_desc; // keep a host copy for state management

  Real *d_data = nullptr;
  AccumulatorType *d_res = nullptr;

  Tensor<Real> host_input;
  int32_t num_slices_total = 0;
  size_t n_elems = 0;
  int32_t n_iterations = Config::STENCIL_ITERS_DEFAULT;
  std::string file_prefix{"./data/"};
  Real scaling_factor = static_cast<Real>(1.0);

public:
  OzakiRunner(const Tensor<Real> &input, const Real a) : host_input(input), n_elems(input.size()) {
    allocate_resources();
    compute_dynamic_range();
  }

  OzakiRunner(const Tensor<Real> &input, const int32_t iterations, const Real a)
      : host_input(input), n_elems(input.size()), n_iterations(iterations), scaling_factor(a) {
    allocate_resources();
    compute_dynamic_range();
  }
  OzakiRunner(const Tensor<Real> &input, const int32_t iterations, const int32_t extra_msb, const int32_t extra_lsb,
              const Real a)
      : host_input(input), n_elems(input.size()), n_iterations(iterations), scaling_factor(a) {
    allocate_resources();
    compute_dynamic_range(extra_msb, extra_lsb);
  }
  OzakiRunner(const Tensor<Real> &input, const int32_t iterations, const int32_t extra_msb, const int32_t extra_lsb,
              const std::string &prefix, const Real a)
      : host_input(input), n_elems(input.size()), n_iterations(iterations), file_prefix(prefix), scaling_factor(a) {
    allocate_resources();
    compute_dynamic_range(extra_msb, extra_lsb);
  }

  ~OzakiRunner() {
    cudaFree(d_slices_1);
    cudaFree(d_slices_2);
    cudaFree(d_desc);
    cudaFree(d_data);
    cudaFree(d_res);
  }

  void allocate_resources() {
    NVTX_PUSH("CUDA Alloc & Upload");
    // Allocate the descriptor struct
    CUDA_CHECK(cudaMalloc(&d_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>)));

    CUDA_CHECK(cudaMalloc(&d_data, n_elems * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_res, n_elems * sizeof(AccumulatorType)));
    CUDA_CHECK(cudaMemcpy(d_data, host_input.data.data(), n_elems * sizeof(Real), cudaMemcpyHostToDevice));
    NVTX_POP();
  }

  void compute_dynamic_range(const int32_t extra_msb = 0, const int32_t extra_lsb = 0) {
    NVTX_PUSH("Compute Dyn Range");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Initialize fresh descriptor on host and copy to device
    h_desc = ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>();
    CUDA_CHECK(cudaMemcpy(d_desc, &h_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>),
                          cudaMemcpyHostToDevice));

    const int32_t nThreads = Config::RDC_THREADS;
    const int32_t nBlocks = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / Config::RDC_THREADS;

    ozigpu::dyn_range<<<nBlocks, nThreads>>>(d_desc, d_data, n_elems);

    // Read back results
    CUDA_CHECK(cudaMemcpy(&h_desc, d_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>),
                          cudaMemcpyDeviceToHost));

    h_desc.e_max += extra_msb;
    h_desc.e_min -= extra_lsb;

    // Calculate total slices based on range
    using TargetType = int8_t; // We are slicing to int8
    num_slices_total = 2 + (h_desc.e_max - h_desc.e_min) / (CHAR_BIT * sizeof(TargetType) - 1);

    // Update descriptor with calculated slice count and modified exponents
    h_desc.n_slices_input = num_slices_total;

    // Send updated config back to device
    CUDA_CHECK(cudaMemcpy(d_desc, &h_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_slices_1, num_slices_total * n_elems * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_slices_2, num_slices_total * n_elems * sizeof(int8_t)));
    NVTX_POP();
  }

  void prepare_slices() {
    NVTX_PUSH("Data Slicing");
    const int32_t nThreads = std::min(static_cast<size_t>(Config::RDC_THREADS), n_elems / 4);
    const int32_t nBlocks = ((n_elems / 4) + nThreads - 1) / nThreads;

    ozigpu::slice<ozigpu::OzakiFP_Int8<Real>><<<nBlocks, nThreads>>>(d_slices_1, d_data, d_desc, n_elems);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(
        cudaMemcpy(d_slices_2, d_slices_1, num_slices_total * n_elems * sizeof(int8_t), cudaMemcpyDeviceToDevice));
    NVTX_POP();
  }

  std::vector<AccumulatorType> run_experiment(int32_t active_slices, const ozi::Stencil<int8_t, 3, 3> &stencil_gpu) {
    NVTX_PUSH(std::format("Run {} Slices", active_slices).c_str());

    // Update descriptor for this specific run config
    h_desc.n_slices_input = active_slices;
    // Set a threshold for overflow detection.
    h_desc.shift_required = 0;
    h_desc.carry_slice_index = 0;
    h_desc.overflow_danger_threshold =
        (std::numeric_limits<int8_t>::max() / static_cast<int8_t>(stencil_gpu.template l1_norm<double>())) - 1;
    h_desc.just_shifted = 1;
    h_desc.n_shifts = 0;

    CUDA_CHECK(cudaMemcpy(d_desc, &h_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>),
                          cudaMemcpyHostToDevice));

    prepare_slices();
    invalidate_caches();

    const dim3 block(Config::STENCIL_BLOCK_DIM_X, Config::STENCIL_BLOCK_DIM_Y);
    const dim3 grid((host_input.width() + block.x - 1) / block.x, (host_input.height() + block.y - 1) / block.y);

    NVTX_PUSH("Integration Loop");
    for (size_t t = 0; t < n_iterations; ++t) {
      // 1. Compute Stencil
      // Reads d_slices_1 (Input), Writes d_slices_2 (Output)
      ozigpu::stencil_v1<ozigpu::OzakiFP_Int8<Real>, 3, 3>
          <<<grid, block>>>(d_slices_2, d_slices_1, stencil_gpu, {host_input.width(), host_input.height()}, d_desc);

      // 2. Check for overflow and rotate slices ONLY if there is a next step
      if (t < n_iterations - 1) {
        ozigpu::update_ozaki_desc<<<1, 32>>>(d_desc);
      }

      // 3. Swap buffers so d_slices_1 becomes the input for next step (or final result)
      std::swap(d_slices_1, d_slices_2);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    NVTX_POP();

    const int32_t nThreadsMend = std::min(static_cast<size_t>(Config::RDC_THREADS), n_elems / 4);
    const int32_t nBlocksMend = ((n_elems / 4) + nThreadsMend - 1) / nThreadsMend;

    const int32_t nThreadsScale = std::min(static_cast<size_t>(Config::RDC_THREADS), n_elems);
    const int32_t nBlocksScale = (n_elems + nThreadsMend - 1) / nThreadsMend;

    NVTX_PUSH("Mending & Download")
    ozigpu::mend<ozigpu::OzakiFP_Int8<Real>><<<nBlocksMend, nThreadsMend>>>(d_res, d_slices_1, d_desc, n_elems);
    CUDA_CHECK(cudaMemcpy(&h_desc, d_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>),
                          cudaMemcpyDeviceToHost));
    ozigpu::scale<<<nBlocksScale, nThreadsScale>>>(d_res,
                                                   ozigpu::total_integration_scale_factor<ozigpu::OzakiFP_Int8<Real>>(
                                                       this->scaling_factor, n_iterations, h_desc.n_shifts),
                                                   n_elems);
    std::cout << std::format("\t{} slices: {} shifts\n", active_slices, h_desc.n_shifts);

    std::vector<AccumulatorType> gpu_result(n_elems);
    CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_res, n_elems * sizeof(AccumulatorType), cudaMemcpyDeviceToHost));
    NVTX_POP();
    NVTX_PUSH("Storing Result");
    write_tensor<AccumulatorType>(std::format("{}_oz_{}.dat", file_prefix, active_slices),
                                  Tensor<AccumulatorType>{host_input.shape, gpu_result});
    NVTX_POP();
    NVTX_POP();

    return gpu_result;
  }

  int get_max_slices() const { return num_slices_total; }
};

int main(int argc, char *argv[]) {
  std::string output_dir;
  std::string output_prefix;
  std::string input_file;
  int32_t n_iters;
  bool run_analysis = false;
  std::vector<int32_t> stencil_args;

  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help,h", "Produce help message")
    ("dir,d", po::value<std::string>(&output_dir)->default_value("./data/"), "Directory where output files will be created")
    ("prefix,p", po::value<std::string>(&output_prefix), "Output file prefix")
    ("analysis,A", po::bool_switch(&run_analysis)->default_value(false), "Run dynamic range analysis")
    ("iters,N", po::value<int32_t>(&n_iters)->default_value(Config::STENCIL_ITERS_DEFAULT), "Number of stencil loop iterations")
    ("input,i", po::value<std::string>(&input_file)->required(), "Path to input file")
    ("stencil,S", po::value<std::vector<int32_t>>(&stencil_args)->multitoken(), "List of int8_t coefficients (e.g. -S 32 -56 ...)");
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
    NVTX_PUSH("Initialization");
    Tensor<Real> input = read_tensor<Real>(input_file);

    auto [heat, heat_ozaki, scaling_factor] = create_stencils(stencil_args);
    NVTX_POP();

    if (run_analysis) {
      std::ofstream analysisf{std::format("{}{}_dyn.csv", output_dir, output_prefix), std::ios::out};
      RangeAnalyzer::analyze(input, heat, n_iters, analysisf);
      std::cout << "Analysis complete.\n";
    }

    std::cout << "Running solvers...\n";
    std::ofstream errf{std::format("{}{}_err.csv", output_dir, output_prefix), std::ios::out};

    auto gold_result = ReferenceSolver::compute<long double>(input, heat, "Gold (long double)", n_iters);

    errf << "Type,Iterations,Slices,Error\n";

    {
      auto res = ReferenceSolver::compute<float>(input, heat, "Ref (float)", n_iters);
      write_tensor(std::format("{}{}_ref_float.dat", output_dir, output_prefix), Tensor<float>{input.shape, res});
      long double mse = calculate_rmse(res, gold_result);
      errf << std::format("float_ref,{},0,{}\n", n_iters, mse);
    }

    {
      auto res = ReferenceSolver::compute<double>(input, heat, "Ref (double)", n_iters);
      write_tensor(std::format("{}{}_ref_double.dat", output_dir, output_prefix), Tensor<double>{input.shape, res});
      long double mse = calculate_rmse(res, gold_result);
      errf << std::format("double_ref,{},0,{}\n", n_iters, mse);
    }

    OzakiRunner runner(input, n_iters, 0, 8, std::format("{}{}", output_dir, output_prefix), scaling_factor);

    NVTX_PUSH("Ozaki Loop");
    for (int32_t s = 1; s < runner.get_max_slices(); ++s) {
      auto gpu_raw = runner.run_experiment(s, heat_ozaki);
      long double error = calculate_rmse(gpu_raw, gold_result);
      errf << std::format("ozaki,{},{},{}\n", n_iters, s, error);
    }
    NVTX_POP();

  } catch (const std::exception &e) {
    std::cerr << std::format("Execution error: {}\n", e.what());
    return 1;
  }

  return 0;
}
