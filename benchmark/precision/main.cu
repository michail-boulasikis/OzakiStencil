
#include "ozaki/gpu/gpu_checks.cuh"
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
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <ozaki/cpu/ozi.hpp>
#include <ozaki/gpu/gpu_checks.cuh>
#include <ozaki/gpu/ozi_stencil_plain.cuh>
#include <ozaki/utils/tensor.hpp>

#ifdef __GNUC__
#include <quadmath.h>
#endif

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using Real = double;
using AccumulatorType = __float128;
using RealStencil = ozi::Stencil<Real, 3, 3>;
using IntStencil = ozi::Stencil<int8_t, 3, 3>;

struct Config {
#ifndef RDC_THREADS_PER_BLOCK
  static constexpr int RDC_THREADS = 512;
#else
  static constexpr int RDC_THREADS = RDC_THREADS_PER_BLOCK;
#endif
  static constexpr int STENCIL_ITERS_DEFAULT = 4096;
  static constexpr size_t STENCIL_BLOCK_DIM_X = 32;
  static constexpr size_t STENCIL_BLOCK_DIM_Y = 8;
};

// Stencil configurations for testing
struct StencilConfig {
  std::string name;
  std::array<int8_t, 9> coeffs_oz;
};

const std::array<StencilConfig, 4> STENCILS = {{{"Heat", {{0, 3, 0, 3, 4, 3, 0, 3, 0}}},
                                                {"AdvectionDiffusion", {{0, 11, 0, 11, 15, 12, 0, 12, 0}}},
                                                {"BoxOnes", {{1, 1, 1, 1, 1, 1, 1, 1, 1}}},
                                                {"BoxRandom", {{-3, 2, 1, -8, 4, -5, -1, -1, 9}}}}};

// Timesteps to test (now configured via command line)

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

template <typename T_Test, typename T_Ref>
long double calculate_rmse(const std::vector<T_Test> &test, const std::vector<T_Ref> &ref) {
  if (test.size() != ref.size()) {
    throw std::runtime_error("Size mismatch in MSE calculation");
  }
#ifdef __GNUC__
  __float128 err = 0.0;
  for (size_t i = 0; i < test.size(); ++i) {
    __float128 diff = static_cast<__float128>(test[i]) - static_cast<__float128>(ref[i]);
    err += diff * diff;
  }
  return static_cast<long double>(sqrtq(err / static_cast<__float128>(test.size())));
#else
  long double err = 0.0;
  for (size_t i = 0; i < test.size(); ++i) {
    long double diff = static_cast<long double>(test[i]) - static_cast<long double>(ref[i]);
    err += diff * diff;
  }
  return std::sqrt(err / static_cast<long double>(test.size()));
#endif
}

std::tuple<RealStencil, IntStencil, Real> create_stencils(const std::array<int8_t, 9> &coeffs_oz) {
  const IntStencil heat_ozaki{coeffs_oz, {1, 1}};
  double l1 = heat_ozaki.template l1_norm<double>();
  Real l1_norm = (l1 > 1e-9) ? static_cast<Real>(l1) : static_cast<Real>(1.0);
  Real scaling_factor = (l1 > 1e-9) ? (static_cast<Real>(1.0) / static_cast<Real>(l1)) : static_cast<Real>(1.0);

  std::array<Real, 9> coeffs;
  for (size_t i = 0; i < 9; ++i) {
    coeffs[i] = static_cast<Real>(coeffs_oz[i]) * scaling_factor;
  }

  const RealStencil heat{coeffs, {1, 1}};
  return {heat, heat_ozaki, l1_norm};
}

// Computes S^3 by correlating the 3x3 stencil S three times to produce a 7x7 stencil
// This represents applying S three times in sequence: 3+3-1 = 5, then 5+3-1 = 7
ozi::Stencil<int8_t, 7, 7> compute_stencil_cubed(const ozi::Stencil<int8_t, 3, 3> &s) {
  auto s2 = ozi::correlate(s, s);
  auto s3 = ozi::correlate(s2, s);
  return s3;
}

template <typename ComputeType>
static ozi::Stencil<ComputeType, 3, 3> make_typed_stencil(const std::array<int8_t, 9> &coeffs_oz, double l1_norm) {
  ozi::Stencil<ComputeType, 3, 3> result;
  result.offsets = {1, 1};

  ComputeType l1_hp = static_cast<ComputeType>(l1_norm);

  for (size_t i = 0; i < coeffs_oz.size(); ++i) {
    result.coeffs[i] = static_cast<ComputeType>(coeffs_oz[i]) / l1_hp;
  }

  return result;
}

struct ReferenceSolver {
  template <typename ComputeType>
  static std::vector<ComputeType> compute(const Tensor<Real> &input, const std::array<int8_t, 9> &coeffs_oz,
                                          double l1_norm, const std::string &label = "Reference",
                                          const int32_t iterations = Config::STENCIL_ITERS_DEFAULT) {
    NVTX_PUSH(std::format("{} Solver", label).c_str());

    std::vector<ComputeType> u_curr(input.data.size());
    std::transform(input.data.begin(), input.data.end(), u_curr.begin(),
                   [](Real x) { return static_cast<ComputeType>(x); });

    std::vector<ComputeType> u_next = u_curr;

    // Use the high-precision coefficient computation helper
    auto stencil_typed = make_typed_stencil<ComputeType>(coeffs_oz, l1_norm);

    for (size_t i = 0; i < iterations + 1; ++i) {
      stencil_typed.template apply_to<ComputeType>(u_curr, u_next, {input.width(), input.height()});
      std::swap(u_curr, u_next);
    }

    NVTX_POP();
    return u_curr;
  }
};

// Base class for Ozaki stencil runners
class OzakiRunnerBase {
protected:
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

  ozi::Stencil<int8_t, 3, 3> stencil_3x3;

public:
  OzakiRunnerBase(const Tensor<Real> &input, const int32_t iterations, const int32_t extra_msb, const int32_t extra_lsb,
                  const std::string &prefix, const Real a, const ozi::Stencil<int8_t, 3, 3> &stencil)
      : host_input(input), n_elems(input.size()), n_iterations(iterations), file_prefix(prefix), scaling_factor(a),
        stencil_3x3(stencil) {
    allocate_resources();
    compute_dynamic_range(extra_msb, extra_lsb);
  }

  virtual ~OzakiRunnerBase() {
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

    h_desc = ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>();
    CUDA_CHECK(cudaMemcpy(d_desc, &h_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>),
                          cudaMemcpyHostToDevice));

    const int32_t nThreads = Config::RDC_THREADS;
    const int32_t nBlocks = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / Config::RDC_THREADS;

    ozigpu::dyn_range<<<nBlocks, nThreads>>>(d_desc, d_data, n_elems);

    CUDA_CHECK(cudaMemcpy(&h_desc, d_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>),
                          cudaMemcpyDeviceToHost));

    h_desc.e_max += extra_msb;
    h_desc.e_min -= extra_lsb;

    using TargetType = int8_t;
    num_slices_total = 2 + (h_desc.e_max - h_desc.e_min) / (CHAR_BIT * sizeof(TargetType) - 1);

    h_desc.n_slices_input = num_slices_total;

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

  // Pure virtual function - implemented differently by subclasses
  virtual std::vector<AccumulatorType> run_experiment(int32_t active_slices, const std::string &output_path,
                                                      const bool save_tensors) = 0;

  int get_max_slices() const { return num_slices_total; }
};

// Direct runner using stencil_v1 kernel
class OzakiRunnerDirect : public OzakiRunnerBase {
public:
  using OzakiRunnerBase::OzakiRunnerBase;

  std::vector<AccumulatorType> run_experiment(int32_t active_slices, const std::string &output_path,
                                              const bool save_tensors) override {
    NVTX_PUSH(std::format("Run {} Slices (Direct)", active_slices).c_str());

    h_desc.n_slices_input = active_slices;
    h_desc.shift_required = 0;
    h_desc.carry_slice_index = 0;
    h_desc.overflow_danger_threshold =
        (std::numeric_limits<int8_t>::max() / static_cast<int8_t>(stencil_3x3.template l1_norm<double>())) - 1;
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
      ozigpu::stencil_v1<ozigpu::OzakiFP_Int8<Real>, 3, 3>
          <<<grid, block>>>(d_slices_2, d_slices_1, stencil_3x3, {host_input.width(), host_input.height()}, d_desc);

      if (t < n_iterations - 1) {
        ozigpu::update_ozaki_desc<<<1, 32>>>(d_desc);
      }

      std::swap(d_slices_1, d_slices_2);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    NVTX_POP();

    const int32_t nThreadsMend = std::min(static_cast<size_t>(Config::RDC_THREADS), n_elems / 4);
    const int32_t nBlocksMend = ((n_elems / 4) + nThreadsMend - 1) / nThreadsMend;

    const int32_t nThreadsScale = std::min(static_cast<size_t>(Config::RDC_THREADS), n_elems);
    const int32_t nBlocksScale = (n_elems + nThreadsScale - 1) / nThreadsScale;

    NVTX_PUSH("Mending & Download")
    ozigpu::mend<ozigpu::OzakiFP_Int8<Real>, AccumulatorType>
        <<<nBlocksMend, nThreadsMend>>>(d_res, d_slices_1, d_desc, n_elems);
    CUDA_CHECK(cudaMemcpy(&h_desc, d_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>),
                          cudaMemcpyDeviceToHost));
#ifdef __GNUC__
    __float128 factor = ozigpu::total_integration_scale_factor_quad<ozigpu::OzakiFP_Int8<Real>>(
        this->scaling_factor, n_iterations, h_desc.n_shifts);
    ozigpu::scale<AccumulatorType, __float128><<<nBlocksScale, nThreadsScale>>>(d_res, factor, n_elems);
#else
    long double factor = total_integration_scale_factor_quad<ozigpu::OzakiFP_Int8<Real>>(this->scaling_factor,
                                                                                         n_iterations, h_desc.n_shifts);
    ozigpu::scale<AccumulatorType, long double><<<nBlocksScale, nThreadsScale>>>(d_res, factor, n_elems);
#endif

    std::vector<AccumulatorType> gpu_result(n_elems);
    CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_res, n_elems * sizeof(AccumulatorType), cudaMemcpyDeviceToHost));
    NVTX_POP();
    if (save_tensors) {
      NVTX_PUSH("Storing Result");
      write_tensor<AccumulatorType>(output_path, Tensor<AccumulatorType>{host_input.shape, gpu_result});
      NVTX_POP();
    }
    NVTX_POP();

    return gpu_result;
  }
};

int main(int argc, char *argv[]) {
  std::string output_dir;
  std::string output_prefix;
  std::string input_file;
  int32_t slice_limit;
  std::vector<size_t> timesteps;
  bool compute_float_ref = false;
  bool compute_double_ref = false;
  bool compute_longdouble_ref = false;
  bool save_tensors = false;

  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help,h", "Produce help message")
    ("dir", po::value<std::string>(&output_dir)->default_value("./data/"),
     "Directory where output tensor files will be created")
    ("prefix,p", po::value<std::string>(&output_prefix)->default_value("precision"),
     "Output file prefix")
    ("input,i", po::value<std::string>(&input_file)->required(), "Path to input tensor file")
    ("slice-limit,s", po::value<int32_t>(&slice_limit)->default_value(24),
     "Maximum slice count to test (starts from 2).")
    ("timesteps,T", po::value<std::vector<size_t>>(&timesteps)->multitoken(),
     "Timesteps to test (e.g., -T 10 100 1000)")
    ("save-tensors,S", "Save every output tensor for further analysis")
    ("float-ref,f", "Compute float reference solution")
    ("double-ref,d", "Compute double reference solution")
    ("longdouble-ref,l", "Compute long double reference solution")
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

    // Set reference computation flags
    compute_float_ref = vm.count("float-ref") > 0;
    compute_double_ref = vm.count("double-ref") > 0;
    compute_longdouble_ref = vm.count("longdouble-ref") > 0;
    save_tensors = vm.count("save-tensors") > 0;

    // Default timesteps if not provided
    if (timesteps.empty()) {
      timesteps = {1000};
    }

  } catch (const po::error &e) {
    std::cerr << "Command Line Error: " << e.what() << "\n";
    std::cerr << "Use -h or --help to see usage.\n";
    return 1;
  }

  try {
    NVTX_PUSH("Initialization");
    Tensor<Real> input = read_tensor<Real>(input_file);
    NVTX_POP();

    // CSV header to stdout
    std::cout << "stencil,type,slice_count,timesteps,error\n";

    for (const auto &stencil_config : STENCILS) {
      NVTX_PUSH(stencil_config.name.c_str());

      // Create stencils once per configuration
      auto [heat, heat_ozaki, l1_norm] = create_stencils(stencil_config.coeffs_oz);

      for (size_t timestep : timesteps) {
        NVTX_PUSH(std::format("{} Timesteps", timestep).c_str());

        // Compute gold reference solution (highest precision available)
        // Use integer coefficients to avoid double-precision rounding errors
#ifdef __GNUC__
        auto gold_result = ReferenceSolver::compute<__float128>(
            input, stencil_config.coeffs_oz, l1_norm, std::format("{} Gold (fp128)", stencil_config.name), timestep);
        if (save_tensors)
          write_tensor(
              std::format("{}{}_{}_{}_gold_fp128.dat", output_dir, output_prefix, stencil_config.name, timestep),
              Tensor<__float128>{input.shape, gold_result});
#else
        auto gold_result =
            ReferenceSolver::compute<long double>(input, stencil_config.coeffs_oz, l1_norm,
                                                  std::format("{} Gold (long double)", stencil_config.name), timestep);
        if (save_tensors)
          write_tensor(
              std::format("{}{}_{}_{}_gold_longdouble.dat", output_dir, output_prefix, stencil_config.name, timestep),
              Tensor<long double>{input.shape, gold_result});
#endif

        // Compute float reference (if requested)
        if (compute_float_ref) {
          auto res = ReferenceSolver::compute<float>(input, stencil_config.coeffs_oz, l1_norm,
                                                     std::format("{} Ref (float)", stencil_config.name), timestep);
          if (save_tensors)
            write_tensor(
                std::format("{}{}_{}_{}_ref_float.dat", output_dir, output_prefix, stencil_config.name, timestep),
                Tensor<float>{input.shape, res});
          long double error = calculate_rmse(res, gold_result);
          std::cout << std::format("{},float_ref,0,{},{}\n", stencil_config.name, timestep, error);
        }

        // Compute double reference (if requested)
        if (compute_double_ref) {
          auto res = ReferenceSolver::compute<double>(input, stencil_config.coeffs_oz, l1_norm,
                                                      std::format("{} Ref (double)", stencil_config.name), timestep);
          if (save_tensors)
            write_tensor(
                std::format("{}{}_{}_{}_ref_double.dat", output_dir, output_prefix, stencil_config.name, timestep),
                Tensor<double>{input.shape, res});
          long double error = calculate_rmse(res, gold_result);
          std::cout << std::format("{},double_ref,0,{},{}\n", stencil_config.name, timestep, error);
        }

        // Compute long double reference (if requested)
        if (compute_longdouble_ref) {
          auto res =
              ReferenceSolver::compute<long double>(input, stencil_config.coeffs_oz, l1_norm,
                                                    std::format("{} Ref (long double)", stencil_config.name), timestep);
          if (save_tensors)
            write_tensor(
                std::format("{}{}_{}_{}_ref_longdouble.dat", output_dir, output_prefix, stencil_config.name, timestep),
                Tensor<long double>{input.shape, res});
          long double error = calculate_rmse(res, gold_result);
          std::cout << std::format("{},longdouble_ref,0,{},{}\n", stencil_config.name, timestep, error);
        }

        // Create runner for this timestep (choose implementation based on flag)
        std::unique_ptr<OzakiRunnerBase> runner;
        std::string runner_type;
        runner = std::make_unique<OzakiRunnerDirect>(input, timestep, 0, 8, output_dir, l1_norm, heat_ozaki);
        runner_type = "ozaki_direct";

        // Test slice counts from 2 to slice_limit
        for (int32_t s = 2; s <= slice_limit && s < runner->get_max_slices(); ++s) {
          std::string output_path =
              std::format("{}{}_{}_{}_oz_{}.dat", output_dir, output_prefix, stencil_config.name, timestep, s);
          auto gpu_result = runner->run_experiment(s, output_path, save_tensors);
          NVTX_PUSH("Error Calculation");
          long double error = calculate_rmse(gpu_result, gold_result);

          // Output CSV row: stencil,type,slice_count,timesteps,error
          std::cout << std::format("{},{},{},{},{}\n", stencil_config.name, runner_type, s, timestep, error);
          NVTX_POP();
        }

        NVTX_POP();
      }

      NVTX_POP();
    }

  } catch (const std::exception &e) {
    std::cerr << std::format("Execution error: {}\n", e.what());
    return 1;
  }

  return 0;
}
