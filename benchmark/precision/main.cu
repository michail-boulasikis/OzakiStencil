#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <ozaki/core/stencil.hpp>
#include <ozaki/cpu/ozi.hpp>
#include <ozaki/gpu/ozi.cuh>

// --- Configuration & Constants ---
struct Config {
#ifndef RDC_THREADS_PER_BLOCK
  static constexpr int RDC_THREADS = 512;
#else
  static constexpr int RDC_THREADS = RDC_THREADS_PER_BLOCK;
#endif
  static constexpr int STENCIL_ITERS = 512;
  static constexpr size_t STENCIL_BLOCK_DIM_X = 16;
  static constexpr size_t STENCIL_BLOCK_DIM_Y = 16;
};

using Real = double;

// --- Helper Structures ---

struct Tensor {
  std::vector<size_t> shape;
  std::vector<Real> data;

  size_t size() const { return data.size(); }
  size_t width() const { return shape.empty() ? 0 : shape[0]; }
  size_t height() const { return shape.size() > 1 ? shape[1] : 1; }
};

// --- IO Utilities ---

Tensor read_tensor(const std::string &filepath) {
  std::ifstream f(filepath, std::ios::in | std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("Failed to open input file: " + filepath);
  }

  size_t n_dims;
  f.read(reinterpret_cast<char *>(&n_dims), sizeof(size_t));

  std::vector<size_t> shape(n_dims);
  f.read(reinterpret_cast<char *>(shape.data()), n_dims * sizeof(size_t));

  size_t n_elems = 1;
  for (const auto &dim : shape)
    n_elems *= dim;

  std::vector<Real> data(n_elems);
  f.read(reinterpret_cast<char *>(data.data()), n_elems * sizeof(Real));

  return {shape, data};
}

// --- Gold Standard (CPU Reference) ---

class GoldStandard {
public:
  static std::vector<long double> compute(const Tensor &input, const ozi::Stencil<Real, 3, 3> &stencil_def) {
    NVTX_PUSH("Gold Standard");

    // Convert input to high precision
    std::vector<long double> u_curr(input.data.size());
    std::transform(input.data.begin(), input.data.end(), u_curr.begin(),
                   [](Real x) { return static_cast<long double>(x); });

    std::vector<long double> u_next = u_curr; // Copy for double buffering

    // Convert stencil coeffs to high precision
    ozi::Stencil<long double, 3, 3> heat_high;
    heat_high.offsets = stencil_def.offsets;
    for (size_t i = 0; i < stencil_def.coeffs.size(); ++i) {
      heat_high.coeffs[i] = static_cast<long double>(stencil_def.coeffs[i]);
    }

    // Run CPU Stencil
    for (size_t i = 0; i < Config::STENCIL_ITERS; ++i) {
      heat_high.template apply_to<long double>(u_curr, u_next, {input.width(), input.height()});
      std::swap(u_curr, u_next);
    }

    NVTX_POP();
    return u_curr; // Result is in u_curr after swap
  }
};

// --- GPU Runner (Ozaki Framework) ---

class OzakiRunner {
  int8_t *d_slices_1 = nullptr, *d_slices_2 = nullptr;
  int32_t *d_dyn_range = nullptr;
  Real *d_data = nullptr, *d_res = nullptr;

  Tensor host_input;
  int32_t num_slices_total = 0;
  size_t n_elems = 0;

public:
  OzakiRunner(const Tensor &input) : host_input(input), n_elems(input.size()) {
    allocate_resources();
    compute_dynamic_range();
  }

  ~OzakiRunner() {
    cudaFree(d_slices_1);
    cudaFree(d_slices_2);
    cudaFree(d_dyn_range);
    cudaFree(d_data);
    cudaFree(d_res);
  }

  void allocate_resources() {
    NVTX_PUSH("CUDA Alloc & Upload");
    CUDA_CHECK(cudaMalloc(&d_dyn_range, 64 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_data, n_elems * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_res, n_elems * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(d_data, host_input.data.data(), n_elems * sizeof(Real), cudaMemcpyHostToDevice));
    NVTX_POP();
  }

  void compute_dynamic_range() {
    NVTX_PUSH("Compute Dyn Range");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Initialize range
    ozigpu::fill<<<1, 1>>>(reinterpret_cast<int2 *>(d_dyn_range), {-1000, +1000}, 1);

    const int32_t nThreads = Config::RDC_THREADS;
    const int32_t nBlocks = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / Config::RDC_THREADS;

    ozigpu::dyn_range<<<nBlocks, nThreads>>>(d_dyn_range, d_data, n_elems);

    int32_t h_dyn_range[64];
    CUDA_CHECK(cudaMemcpy(h_dyn_range, d_dyn_range, 64 * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Calculate required slices based on bit depth
    num_slices_total = 1 + (h_dyn_range[0] - h_dyn_range[1]) / (CHAR_BIT * sizeof(int8_t) - 1);

    // Allocate Slice Memory
    CUDA_CHECK(cudaMalloc(&d_slices_1, num_slices_total * n_elems * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_slices_2, num_slices_total * n_elems * sizeof(int8_t)));
    NVTX_POP();
  }

  void prepare_slices() {
    NVTX_PUSH("Data Slicing");
    const int32_t nThreads = std::min(static_cast<size_t>(Config::RDC_THREADS), n_elems / 4);
    const int32_t nBlocks = ((n_elems / 4) + nThreads - 1) / nThreads;

    ozigpu::slice<ozigpu::OzakiFP_Int8<Real>>
        <<<nBlocks, nThreads>>>(d_slices_1, d_data, d_dyn_range, num_slices_total, n_elems);

    CUDA_CHECK(cudaDeviceSynchronize());
    // Init buffer 2
    CUDA_CHECK(
        cudaMemcpy(d_slices_2, d_slices_1, num_slices_total * n_elems * sizeof(int8_t), cudaMemcpyDeviceToDevice));
    NVTX_POP();
  }

  // Returns MSE error for a specific slice count
  long double run_experiment(int32_t active_slices, const ozi::Stencil<int8_t, 3, 3> &stencil_gpu,
                             const std::vector<long double> &gold_result) {

    NVTX_PUSH(std::format("Run {} Slices", active_slices).c_str());

    // Reset state
    prepare_slices();

    const dim3 block(Config::STENCIL_BLOCK_DIM_X, Config::STENCIL_BLOCK_DIM_Y);
    const dim3 grid((host_input.width() + block.x - 1) / block.x, (host_input.height() + block.y - 1) / block.y);

    // Run Stencil
    NVTX_PUSH("Integration Loop");
    for (size_t t = 0; t < Config::STENCIL_ITERS; ++t) {
      ozigpu::stencil_v1<ozigpu::OzakiFP_Int8<Real>, 3, 3><<<grid, block>>>(
          d_slices_2, d_slices_1, stencil_gpu, {host_input.width(), host_input.height()}, active_slices);
      std::swap(d_slices_1, d_slices_2);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    NVTX_POP();

    // Mend (Reconstruct)
    const int32_t nThreadsMend = std::min(static_cast<size_t>(Config::RDC_THREADS), n_elems / 4);
    const int32_t nBlocksMend = ((n_elems / 4) + nThreadsMend - 1) / nThreadsMend;

    ozigpu::mend<ozigpu::OzakiFP_Int8<Real>>
        <<<nBlocksMend, nThreadsMend>>>(d_res, d_slices_2, d_dyn_range, active_slices, n_elems);

    // Download result
    std::vector<Real> gpu_result(n_elems);
    CUDA_CHECK(cudaMemcpy(gpu_result.data(), d_res, n_elems * sizeof(Real), cudaMemcpyDeviceToHost));
    NVTX_POP();

    return calculate_mse(gpu_result, gold_result);
  }

  int get_max_slices() const { return num_slices_total; }

private:
  long double calculate_mse(const std::vector<Real> &gpu_res, const std::vector<long double> &gold) {
    long double err = 0.0;
    for (size_t i = 0; i < n_elems; ++i) {
      long double diff = static_cast<long double>(gpu_res[i]) - gold[i];
      err += diff * diff;
    }
    return err / static_cast<long double>(n_elems);
  }
};

// --- Main ---

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << std::format("Usage: {} <input_tensor_file>\n", argv[0]);
    return 1;
  }

  try {
    // 1. Load Data
    NVTX_PUSH("Initialization");
    Tensor input = read_tensor(argv[1]);
    // std::cout << std::format("Loaded tensor: {}x{}\n", input.width(), input.height());

    // 2. Define Stencils
    const Real r = 3.0 / 16.0;
    const std::array<Real, 9> coeffs{0.0, r, 0.0, r, 1.0 - 4.0 * r, r, 0.0, r, 0.0};
    const std::array<int8_t, 9> coeffs_oz{0, 3 * 8, 0, 3 * 8, 4 * 8, 3 * 8, 0, 3 * 8, 0};

    const ozi::Stencil<Real, 3, 3> heat{coeffs, {1, 1}};
    const ozi::Stencil<int8_t, 3, 3> heat_ozaki{coeffs_oz, {1, 1}};
    NVTX_POP();

    // 3. Compute Gold Standard
    auto gold_result = GoldStandard::compute(input, heat);

    // 4. Run GPU Experiments
    OzakiRunner runner(input);

    std::cout << "slices,error\n";
    NVTX_PUSH("Ozaki Loop");
    for (int32_t s = 1; s < runner.get_max_slices(); ++s) {
      long double error = runner.run_experiment(s, heat_ozaki, gold_result);
      std::cout << std::format("{},{}\n", s, error);
    }
    NVTX_POP();

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
