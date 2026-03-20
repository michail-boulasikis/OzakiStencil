#include <ozaki/cpu/ozi.hpp>
#include <ozaki/gpu/benchmark_utils.cuh>
#include <ozaki/gpu/gpu_checks.cuh>
#include <ozaki/gpu/ozi.cuh>
#include <ozaki/gpu/ozi_kernels.cuh>
#include <ozaki/gpu/ozi_stencil_conv_v2.cuh>
#include <ozaki/gpu/ozi_stencil_plain.cuh>
#include <ozaki/gpu/stencil_gpu.cuh>
#include <ozaki/utils/stencil_registry.hpp>
#include <ozaki/utils/stencil_spec.hpp>
#include <ozaki/utils/tensor.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include <boost/program_options.hpp>

#ifdef OZAKI_HAVE_FLOAT128
#include <quadmath.h>
#endif

namespace po = boost::program_options;

using Real = double;
using Oz = ozigpu::OzakiFP_Int8<Real>;

// Configuration and result types

struct BenchConfig {
  std::vector<size_t> shape;
  int32_t timesteps = 100;
  int32_t warmup = 10;
  int32_t repeats = 3;
  int32_t slices_override = 0;
  int32_t extra_msb = 0;
  int32_t extra_lsb = 0;
  uint64_t seed = 42;
  bool flush_l2 = true;
  bool ncu_mode = false;
  dim3 block_size{0, 0, 0}; // {0,0,0} = use default
};

struct PhaseTimings {
  std::vector<float> dyn_range_ms;   // one per repeat
  std::vector<float> slice_ms;       // one per repeat
  std::vector<float> integration_ms; // one per repeat
  std::vector<float> mend_ms;        // one per repeat
};

enum class AlgoFlag {
  NaiveFP32 = 1 << 0,
  NaiveFP64 = 1 << 1,
  NaiveFP128 = 1 << 2,
  Direct = 1 << 3,
  ConvStencil = 1 << 4,
};

static int parse_algos(const std::string &arg, int device_cc) {
  int flags = 0;
  std::istringstream ss(arg);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token == "all") {
      flags = 0xFF;
    } else if (token == "naive") {
      flags |= static_cast<int>(AlgoFlag::NaiveFP32) | static_cast<int>(AlgoFlag::NaiveFP64) |
               static_cast<int>(AlgoFlag::NaiveFP128);
    } else if (token == "naive-fp32") {
      flags |= static_cast<int>(AlgoFlag::NaiveFP32);
    } else if (token == "naive-fp64") {
      flags |= static_cast<int>(AlgoFlag::NaiveFP64);
    } else if (token == "naive-fp128") {
      flags |= static_cast<int>(AlgoFlag::NaiveFP128);
    } else if (token == "direct") {
      flags |= static_cast<int>(AlgoFlag::Direct);
    } else if (token == "convstencil") {
      flags |= static_cast<int>(AlgoFlag::ConvStencil);
    } else {
      throw std::invalid_argument("Unknown algorithm: " + token);
    }
  }
  return flags;
}

static bool has_algo(int flags, AlgoFlag a) { return (flags & static_cast<int>(a)) != 0; }

// CUDA event timing helpers

struct CudaEventPair {
  cudaEvent_t start, stop;
  CudaEventPair() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~CudaEventPair() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  CudaEventPair(const CudaEventPair &) = delete;
  CudaEventPair &operator=(const CudaEventPair &) = delete;

  void record_start() { CUDA_CHECK(cudaEventRecord(start)); }
  void record_stop() { CUDA_CHECK(cudaEventRecord(stop)); }
  float elapsed_ms() {
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

// Statistics helpers

static float stat_min(const std::vector<float> &v) { return *std::min_element(v.begin(), v.end()); }

static float stat_median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  if (n % 2 == 0)
    return (v[n / 2 - 1] + v[n / 2]) / 2.0f;
  return v[n / 2];
}

static float stat_max(const std::vector<float> &v) { return *std::max_element(v.begin(), v.end()); }

static double compute_gstpts(const std::vector<size_t> &shape, int32_t timesteps, float time_ms) {
  size_t n = 1;
  for (auto s : shape)
    n *= s;
  return static_cast<double>(n) * timesteps / (time_ms * 1e-3) / 1e9;
}

// Grid/block helpers (same conventions as ozaki_runner.cuh)

template <size_t NDims>
static std::pair<dim3, dim3> make_grid_block(const std::vector<size_t> &shape, const dim3 *block_override = nullptr) {
  dim3 block;
  if (block_override && block_override->x > 0) {
    block = *block_override;
  } else if constexpr (NDims == 1) {
    block = dim3(256);
  } else if constexpr (NDims == 2) {
    block = dim3(32, 8);
  } else {
    block = dim3(8, 8, 4);
  }

  dim3 grid;
  if constexpr (NDims == 1) {
    grid = dim3((shape[0] + block.x - 1) / block.x);
  } else if constexpr (NDims == 2) {
    grid = dim3((shape[1] + block.x - 1) / block.x, (shape[0] + block.y - 1) / block.y);
  } else {
    grid = dim3((shape[2] + block.x - 1) / block.x, (shape[1] + block.y - 1) / block.y,
                (shape[0] + block.z - 1) / block.z);
  }
  return {grid, block};
}

// Naive FP benchmark (templated on Real type)

template <typename FPType, size_t... Dims>
static void launch_naive_kernel(FPType *out, const FPType *in,
                                const typename ozi::Stencil<FPType, Dims...>::shape_type &input_shape, dim3 grid,
                                dim3 block) {
  if constexpr (std::is_same_v<FPType, float>) {
    ozi::stencil_fp32<Dims...><<<grid, block>>>(out, in, input_shape);
  } else if constexpr (std::is_same_v<FPType, double>) {
    ozi::stencil_fp64<Dims...><<<grid, block>>>(out, in, input_shape);
  } else {
#if !defined(__CUDA_ARCH_LIST__) || __CUDA_ARCH_LIST__ >= 1000
    ozi::stencil_fp128<Dims...><<<grid, block>>>(out, in, input_shape);
#endif
  }
}

template <typename FPType, size_t... Dims>
PhaseTimings bench_naive(const ozi::Stencil<CoeffStorageType, Dims...> &coeff_stencil, const std::vector<size_t> &shape,
                         const BenchConfig &cfg, ozigpu::L2CacheInvalidator &cache_inv, const char *label) {
  constexpr size_t ndims = sizeof...(Dims);
  size_t n_elems = 1;
  for (auto s : shape)
    n_elems *= s;

  // Build typed stencil for this FPType
  ozi::Stencil<FPType, Dims...> h_stencil;
  for (size_t i = 0; i < ndims; ++i)
    h_stencil.offsets[i] = coeff_stencil.offsets[i];
  for (size_t i = 0; i < h_stencil.size; ++i)
    h_stencil.coeffs[i] = static_cast<FPType>(coeff_stencil.coeffs[i]);

  // Allocate device memory
  FPType *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n_elems * sizeof(FPType)));
  CUDA_CHECK(cudaMalloc(&d_out, n_elems * sizeof(FPType)));
  CUDA_CHECK(cudaMemcpyToSymbol(ozi::stencil_cmem, &h_stencil, sizeof(h_stencil)));

  // Generate random data on host and upload
  std::vector<FPType> h_data(n_elems);
  std::mt19937_64 rng(cfg.seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (size_t i = 0; i < n_elems; ++i)
    h_data[i] = static_cast<FPType>(dist(rng));
  CUDA_CHECK(cudaMemcpy(d_in, h_data.data(), n_elems * sizeof(FPType), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, n_elems * sizeof(FPType)));

  const dim3 *blk_ptr = (cfg.block_size.x > 0) ? &cfg.block_size : nullptr;
  auto [grid, block] = make_grid_block<ndims>(shape, blk_ptr);

  typename ozi::Stencil<FPType, Dims...>::shape_type input_shape{};
  for (size_t d = 0; d < ndims; ++d)
    input_shape[d] = shape[d];

  PhaseTimings result;

  NVTX_PUSH(label);

  for (int32_t r = 0; r < cfg.repeats; ++r) {
    // Re-upload data for each repeat to get a clean state
    CUDA_CHECK(cudaMemcpy(d_in, h_data.data(), n_elems * sizeof(FPType), cudaMemcpyHostToDevice));

    if (cfg.flush_l2)
      cache_inv.invalidate();

    // Warmup
    NVTX_PUSH("Warmup");
    for (int32_t t = 0; t < cfg.warmup; ++t) {
      launch_naive_kernel<FPType, Dims...>(d_out, d_in, input_shape, grid, block);
      CUDA_CHECK(cudaGetLastError());
      std::swap(d_in, d_out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    NVTX_POP();

    if (cfg.flush_l2)
      cache_inv.invalidate();

    // Measurement
    CudaEventPair ev;
    NVTX_PUSH("Integration");
    ev.record_start();
    for (int32_t t = 0; t < cfg.timesteps; ++t) {
      launch_naive_kernel<FPType, Dims...>(d_out, d_in, input_shape, grid, block);
      CUDA_CHECK(cudaGetLastError());
      std::swap(d_in, d_out);
    }
    ev.record_stop();
    NVTX_POP();

    result.integration_ms.push_back(ev.elapsed_ms());
  }

  NVTX_POP(); // label

  cudaFree(d_in);
  cudaFree(d_out);
  return result;
}

// Ozaki Direct benchmark

static constexpr int RDC_THREADS = 512;

template <int32_t M, size_t... Dims>
PhaseTimings bench_ozaki_direct(const ozi::StencilSlices<ozi::Stencil<int8_t, Dims...>, M> &ss,
                                const std::vector<size_t> &shape, int32_t num_slices_total, HighPrecType scaling_factor,
                                const BenchConfig &cfg, ozigpu::L2CacheInvalidator &cache_inv) {
  constexpr size_t ndims = sizeof...(Dims);
  size_t n_elems = 1;
  for (auto s : shape)
    n_elems *= s;

  std::string nvtx_label = "Ozaki Direct (M=" + std::to_string(M) + ", S=" + std::to_string(num_slices_total) + ")";
  NVTX_PUSH(nvtx_label.c_str());

  // Allocate device resources
  Real *d_data = nullptr;
  int8_t *d_slices_1 = nullptr, *d_slices_2 = nullptr;
  ozigpu::OzakiDescriptor<Oz> *d_desc = nullptr;
  ozigpu::OzakiDescriptor<Oz> h_desc;

  CUDA_CHECK(cudaMalloc(&d_desc, sizeof(ozigpu::OzakiDescriptor<Oz>)));
  CUDA_CHECK(cudaMalloc(&d_data, n_elems * sizeof(Real)));

  // Upload random data
  std::vector<Real> h_data(n_elems);
  std::mt19937_64 rng(cfg.seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (size_t i = 0; i < n_elems; ++i)
    h_data[i] = dist(rng);
  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n_elems * sizeof(Real), cudaMemcpyHostToDevice));

  PhaseTimings result;
  CudaEventPair ev_phase;

  // -- DynRange (with warmup + repetition) --
  NVTX_PUSH("DynRange");
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const int32_t nBlocksDR = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / RDC_THREADS;

  // Warmup
  if (cfg.warmup > 0) {
    h_desc = ozigpu::OzakiDescriptor<Oz>();
    CUDA_CHECK(cudaMemcpy(d_desc, &h_desc, sizeof(h_desc), cudaMemcpyHostToDevice));
    ozigpu::dyn_range<Oz><<<nBlocksDR, RDC_THREADS>>>(d_desc, d_data, n_elems);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Measured repeats
  for (int32_t r = 0; r < cfg.repeats; ++r) {
    h_desc = ozigpu::OzakiDescriptor<Oz>();
    CUDA_CHECK(cudaMemcpy(d_desc, &h_desc, sizeof(h_desc), cudaMemcpyHostToDevice));
    ev_phase.record_start();
    ozigpu::dyn_range<Oz><<<nBlocksDR, RDC_THREADS>>>(d_desc, d_data, n_elems);
    ev_phase.record_stop();
    result.dyn_range_ms.push_back(ev_phase.elapsed_ms());
  }
  NVTX_POP();

  // Download descriptor from last repeat and compute actual slice count
  CUDA_CHECK(cudaMemcpy(&h_desc, d_desc, sizeof(h_desc), cudaMemcpyDeviceToHost));
  h_desc.e_max += cfg.extra_msb;
  h_desc.e_min -= cfg.extra_lsb;

  int32_t actual_slices = num_slices_total;
  if (cfg.slices_override > 0) {
    actual_slices = cfg.slices_override;
  } else {
    actual_slices = 2 + (h_desc.e_max - h_desc.e_min) / (CHAR_BIT * sizeof(int8_t) - 1);
  }

  h_desc.n_slices_input = actual_slices;
  CUDA_CHECK(cudaMemcpy(d_desc, &h_desc, sizeof(h_desc), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_slices_1, actual_slices * n_elems * sizeof(int8_t) + ozigpu::STENCIL_V1_ALLOC_PAD));
  CUDA_CHECK(cudaMalloc(&d_slices_2, actual_slices * n_elems * sizeof(int8_t) + ozigpu::STENCIL_V1_ALLOC_PAD));

  // -- Slice (with warmup + repetition; D2D copy excluded from timing) --
  NVTX_PUSH("Slice");
  const int32_t nThreadsSlice = std::min(static_cast<size_t>(RDC_THREADS), n_elems / 4);
  const int32_t nBlocksSlice = ((n_elems / 4) + nThreadsSlice - 1) / nThreadsSlice;

  // Warmup
  if (cfg.warmup > 0) {
    ozigpu::slice<Oz><<<nBlocksSlice, nThreadsSlice>>>(d_slices_1, d_data, d_desc, n_elems);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Measured repeats
  for (int32_t r = 0; r < cfg.repeats; ++r) {
    ev_phase.record_start();
    ozigpu::slice<Oz><<<nBlocksSlice, nThreadsSlice>>>(d_slices_1, d_data, d_desc, n_elems);
    ev_phase.record_stop();
    result.slice_ms.push_back(ev_phase.elapsed_ms());
  }

  // Set up ping-pong buffer (not part of slice timing - benchmark infrastructure)
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(d_slices_2, d_slices_1, actual_slices * n_elems * sizeof(int8_t), cudaMemcpyDeviceToDevice));
  NVTX_POP();

  // Compute overflow threshold
  double slice0_l1 = ss.slices[0].template l1_norm<double>();
  int32_t l1_int = static_cast<int32_t>(slice0_l1);
  if (l1_int == 0)
    l1_int = 1;

  // Proactive shift schedule
  double log2_sf = (scaling_factor > static_cast<HighPrecType>(1)) ? std::log2(static_cast<double>(scaling_factor)) : 0.0;
  int32_t total_shifts_needed = static_cast<int32_t>(std::ceil(cfg.timesteps * log2_sf / 7.0));

  auto [grid, block] = make_grid_block<ndims>(shape);

  typename ozi::Stencil<int8_t, Dims...>::shape_type input_shape{};
  for (size_t d = 0; d < ndims; ++d)
    input_shape[d] = shape[d];

  for (int32_t r = 0; r < cfg.repeats; ++r) {
    // Reset state for each repeat
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n_elems * sizeof(Real), cudaMemcpyHostToDevice));

    h_desc.n_slices_input = actual_slices;
    h_desc.shift_required = 0;
    h_desc.carry_slice_index = 0;
    h_desc.overflow_danger_threshold = (std::numeric_limits<int8_t>::max() / l1_int) - 1;
    h_desc.just_shifted = 1;
    h_desc.n_shifts = 0;
    CUDA_CHECK(cudaMemcpy(d_desc, &h_desc, sizeof(h_desc), cudaMemcpyHostToDevice));

    // Re-slice
    ozigpu::slice<Oz><<<nBlocksSlice, nThreadsSlice>>>(d_slices_1, d_data, d_desc, n_elems);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_slices_2, d_slices_1, actual_slices * n_elems * sizeof(int8_t), cudaMemcpyDeviceToDevice));

    if (cfg.flush_l2)
      cache_inv.invalidate();

    // Copy stencil slices to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(ozigpu::stencil_v1_cmem, &ss, sizeof(ss)));
    ozigpu::upload_slice_phys_lut(actual_slices);

    // Warmup
    if (cfg.warmup > 0) {
      NVTX_PUSH("Warmup");
      for (int32_t t = 0; t < cfg.warmup; ++t) {
        ozigpu::stencil_v1<Oz, M, Dims...><<<grid, block>>>(d_slices_2, d_slices_1, input_shape, d_desc);
        if (t < cfg.warmup - 1)
          ozigpu::update_ozaki_desc<<<1, 32>>>(d_desc, 0);
        std::swap(d_slices_1, d_slices_2);
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      NVTX_POP();

      // Reset desc state after warmup
      h_desc.n_slices_input = actual_slices;
      h_desc.shift_required = 0;
      h_desc.carry_slice_index = 0;
      h_desc.overflow_danger_threshold = (std::numeric_limits<int8_t>::max() / l1_int) - 1;
      h_desc.just_shifted = 1;
      h_desc.n_shifts = 0;
      CUDA_CHECK(cudaMemcpy(d_desc, &h_desc, sizeof(h_desc), cudaMemcpyHostToDevice));

      // Re-slice again for clean measurement
      ozigpu::slice<Oz><<<nBlocksSlice, nThreadsSlice>>>(d_slices_1, d_data, d_desc, n_elems);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(
          cudaMemcpy(d_slices_2, d_slices_1, actual_slices * n_elems * sizeof(int8_t), cudaMemcpyDeviceToDevice));
    }

    if (cfg.flush_l2)
      cache_inv.invalidate();

    // Measurement
    CudaEventPair ev;
    NVTX_PUSH("Integration");
    ev.record_start();
    for (int32_t t = 0; t < cfg.timesteps; ++t) {
      ozigpu::stencil_v1<Oz, M, Dims...><<<grid, block>>>(d_slices_2, d_slices_1, input_shape, d_desc);
      if (t < cfg.timesteps - 1) {
        int32_t min_shifts_now;
        if (t == cfg.timesteps - 2)
          min_shifts_now = total_shifts_needed;
        else
          min_shifts_now = static_cast<int32_t>(std::ceil((t + 1) * log2_sf / 7.0));
        if (min_shifts_now < 0)
          min_shifts_now = 0;
        ozigpu::update_ozaki_desc<<<1, 32>>>(d_desc, min_shifts_now);
      }
      std::swap(d_slices_1, d_slices_2);
    }
    ev.record_stop();
    NVTX_POP();

    result.integration_ms.push_back(ev.elapsed_ms());
  }

  // -- Mend (with warmup + repetition) --
  NVTX_PUSH("Mend");
  const int32_t nThreadsMend = std::min(static_cast<size_t>(RDC_THREADS), n_elems / 4);
  const int32_t nBlocksMend = ((n_elems / 4) + nThreadsMend - 1) / nThreadsMend;

  // Warmup
  if (cfg.warmup > 0) {
    ozigpu::mend<Oz><<<nBlocksMend, nThreadsMend>>>(d_data, d_slices_1, d_desc, n_elems);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Measured repeats (mend reads slices+desc without modifying them, so repeats are safe)
  for (int32_t r = 0; r < cfg.repeats; ++r) {
    ev_phase.record_start();
    ozigpu::mend<Oz><<<nBlocksMend, nThreadsMend>>>(d_data, d_slices_1, d_desc, n_elems);
    ev_phase.record_stop();
    result.mend_ms.push_back(ev_phase.elapsed_ms());
  }
  NVTX_POP();

  NVTX_POP(); // nvtx_label

  cudaFree(d_data);
  cudaFree(d_slices_1);
  cudaFree(d_slices_2);
  cudaFree(d_desc);
  return result;
}

// Ozaki ConvStencil benchmark (2D 7x7 only)

template <int32_t M>
PhaseTimings bench_ozaki_conv(const ozi::StencilSlices<ozi::Stencil<int8_t, 7, 7>, M> &ss, const int8_t *host_weight,
                              const ozigpu::convstencil::BoundaryGainData &boundary, const std::vector<size_t> &shape,
                              int32_t num_slices_total, HighPrecType scaling_factor, const BenchConfig &cfg,
                              ozigpu::L2CacheInvalidator &cache_inv) {
  size_t n_elems = shape[0] * shape[1];
  const size_t height = shape[0];
  const size_t ldm = shape[1];

  std::string nvtx_label = "Ozaki ConvStencil (M=" + std::to_string(M) + ", S=" + std::to_string(num_slices_total) + ")";
  NVTX_PUSH(nvtx_label.c_str());

  // Allocate device resources
  Real *d_data = nullptr;
  int8_t *d_slices_1 = nullptr, *d_slices_2 = nullptr, *d_weight = nullptr;
  ozigpu::OzakiDescriptor<Oz> *d_desc = nullptr;
  ozigpu::OzakiDescriptor<Oz> h_desc;

  CUDA_CHECK(cudaMalloc(&d_desc, sizeof(ozigpu::OzakiDescriptor<Oz>)));
  CUDA_CHECK(cudaMalloc(&d_data, n_elems * sizeof(Real)));

  constexpr size_t weight_size = M * 2 * ozigpu::convstencil::WT_SIZE;
  CUDA_CHECK(cudaMalloc(&d_weight, weight_size));
  CUDA_CHECK(cudaMemcpy(d_weight, host_weight, weight_size, cudaMemcpyHostToDevice));

  // Upload random data
  std::vector<Real> h_data(n_elems);
  std::mt19937_64 rng(cfg.seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (size_t i = 0; i < n_elems; ++i)
    h_data[i] = dist(rng);
  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n_elems * sizeof(Real), cudaMemcpyHostToDevice));

  PhaseTimings result;
  CudaEventPair ev_phase;

  // -- DynRange (with warmup + repetition) --
  NVTX_PUSH("DynRange");
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const int32_t nBlocksDR = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / RDC_THREADS;

  // Warmup
  if (cfg.warmup > 0) {
    h_desc = ozigpu::OzakiDescriptor<Oz>();
    CUDA_CHECK(cudaMemcpy(d_desc, &h_desc, sizeof(h_desc), cudaMemcpyHostToDevice));
    ozigpu::dyn_range<Oz><<<nBlocksDR, RDC_THREADS>>>(d_desc, d_data, n_elems);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Measured repeats
  for (int32_t r = 0; r < cfg.repeats; ++r) {
    h_desc = ozigpu::OzakiDescriptor<Oz>();
    CUDA_CHECK(cudaMemcpy(d_desc, &h_desc, sizeof(h_desc), cudaMemcpyHostToDevice));
    ev_phase.record_start();
    ozigpu::dyn_range<Oz><<<nBlocksDR, RDC_THREADS>>>(d_desc, d_data, n_elems);
    ev_phase.record_stop();
    result.dyn_range_ms.push_back(ev_phase.elapsed_ms());
  }
  NVTX_POP();

  // Download descriptor from last repeat and compute slice count
  CUDA_CHECK(cudaMemcpy(&h_desc, d_desc, sizeof(h_desc), cudaMemcpyDeviceToHost));
  h_desc.e_max += cfg.extra_msb;
  h_desc.e_min -= cfg.extra_lsb;

  int32_t actual_slices = num_slices_total;
  if (cfg.slices_override > 0) {
    actual_slices = cfg.slices_override;
  } else {
    actual_slices = 2 + (h_desc.e_max - h_desc.e_min) / (CHAR_BIT * sizeof(int8_t) - 1);
  }

  h_desc.n_slices_input = actual_slices;
  CUDA_CHECK(cudaMemcpy(d_desc, &h_desc, sizeof(h_desc), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_slices_1, actual_slices * n_elems * sizeof(int8_t) + ozigpu::STENCIL_V1_ALLOC_PAD));
  CUDA_CHECK(cudaMalloc(&d_slices_2, actual_slices * n_elems * sizeof(int8_t) + ozigpu::STENCIL_V1_ALLOC_PAD));

  // -- Slice (with warmup + repetition; D2D copy excluded from timing) --
  NVTX_PUSH("Slice");
  const int32_t nThreadsSlice = std::min(static_cast<size_t>(RDC_THREADS), n_elems / 4);
  const int32_t nBlocksSlice = ((n_elems / 4) + nThreadsSlice - 1) / nThreadsSlice;

  // Warmup
  if (cfg.warmup > 0) {
    ozigpu::slice<Oz><<<nBlocksSlice, nThreadsSlice>>>(d_slices_1, d_data, d_desc, n_elems);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Measured repeats
  for (int32_t r = 0; r < cfg.repeats; ++r) {
    ev_phase.record_start();
    ozigpu::slice<Oz><<<nBlocksSlice, nThreadsSlice>>>(d_slices_1, d_data, d_desc, n_elems);
    ev_phase.record_stop();
    result.slice_ms.push_back(ev_phase.elapsed_ms());
  }

  // Set up ping-pong buffer (not part of slice timing -- benchmark infrastructure)
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(d_slices_2, d_slices_1, actual_slices * n_elems * sizeof(int8_t), cudaMemcpyDeviceToDevice));
  NVTX_POP();

  // Compute overflow threshold
  double slice0_l1 = ss.slices[0].template l1_norm<double>();
  int32_t l1_int = static_cast<int32_t>(slice0_l1);
  if (l1_int == 0)
    l1_int = 1;

  // Proactive shift schedule
  double log2_sf = (scaling_factor > static_cast<HighPrecType>(1)) ? std::log2(static_cast<double>(scaling_factor)) : 0.0;
  int32_t total_shifts_needed = static_cast<int32_t>(std::ceil(cfg.timesteps * log2_sf / 7.0));

  // ConvStencil grid/block
  constexpr int32_t BLOCK_THREADS = ozigpu::convstencil::NUM_WARPS * WARP_SIZE;
  dim3 block_cv(BLOCK_THREADS);
  dim3 grid_cv((height + ozigpu::convstencil::TILE_SIZE_ROW - 1) / ozigpu::convstencil::TILE_SIZE_ROW,
               (ldm + ozigpu::convstencil::TILE_SIZE_COL - 1) / ozigpu::convstencil::TILE_SIZE_COL);

  cudaFuncSetAttribute(ozigpu::cvstencil_v2<Oz, M>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       ozigpu::convstencil::DYN_SMEM_BYTES);

  ozigpu::convstencil::TmaDescriptorsV2 tma;
  size_t total_height = static_cast<size_t>(actual_slices) * height;
  bool tma_ok = ozigpu::convstencil::create_tma_descriptors_v2(tma, d_slices_1, d_slices_2, total_height, ldm);
  if (!tma_ok) {
    std::cerr << "Skipping cvstencil: TMA requires ldm >= 128 and ldm % 16 == 0\n";
    return {};
  }

  for (int32_t r = 0; r < cfg.repeats; ++r) {
    // Reset state
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n_elems * sizeof(Real), cudaMemcpyHostToDevice));

    h_desc.n_slices_input = actual_slices;
    h_desc.shift_required = 0;
    h_desc.carry_slice_index = 0;
    h_desc.overflow_danger_threshold = (std::numeric_limits<int8_t>::max() / l1_int) - 1;
    h_desc.just_shifted = 1;
    h_desc.n_shifts = 0;
    CUDA_CHECK(cudaMemcpy(d_desc, &h_desc, sizeof(h_desc), cudaMemcpyHostToDevice));

    // Re-slice
    ozigpu::slice<Oz><<<nBlocksSlice, nThreadsSlice>>>(d_slices_1, d_data, d_desc, n_elems);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_slices_2, d_slices_1, actual_slices * n_elems * sizeof(int8_t), cudaMemcpyDeviceToDevice));

    ozigpu::upload_slice_phys_lut(actual_slices);

    if (cfg.flush_l2)
      cache_inv.invalidate();

    // Warmup
    if (cfg.warmup > 0) {
      NVTX_PUSH("Warmup");
      for (int32_t t = 0; t < cfg.warmup; ++t) {
        ozigpu::cvstencil_v2<Oz, M>
            <<<grid_cv, block_cv, ozigpu::convstencil::DYN_SMEM_BYTES>>>(d_slices_2, d_slices_1, d_weight, ldm, height, d_desc, boundary, tma.h_main[0]);
        if (t < cfg.warmup - 1)
          ozigpu::update_ozaki_desc<<<1, 32>>>(d_desc, 0);
        std::swap(d_slices_1, d_slices_2);
        tma.swap();
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      NVTX_POP();

      // Reset desc state after warmup
      h_desc.n_slices_input = actual_slices;
      h_desc.shift_required = 0;
      h_desc.carry_slice_index = 0;
      h_desc.overflow_danger_threshold = (std::numeric_limits<int8_t>::max() / l1_int) - 1;
      h_desc.just_shifted = 1;
      h_desc.n_shifts = 0;
      CUDA_CHECK(cudaMemcpy(d_desc, &h_desc, sizeof(h_desc), cudaMemcpyHostToDevice));

      // Re-slice again for clean measurement
      ozigpu::slice<Oz><<<nBlocksSlice, nThreadsSlice>>>(d_slices_1, d_data, d_desc, n_elems);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(
          cudaMemcpy(d_slices_2, d_slices_1, actual_slices * n_elems * sizeof(int8_t), cudaMemcpyDeviceToDevice));
    }

    if (cfg.flush_l2)
      cache_inv.invalidate();

    // Measurement
    CudaEventPair ev;
    NVTX_PUSH("Integration");
    ev.record_start();
    for (int32_t t = 0; t < cfg.timesteps; ++t) {
      ozigpu::cvstencil_v2<Oz, M><<<grid_cv, block_cv, ozigpu::convstencil::DYN_SMEM_BYTES>>>(d_slices_2, d_slices_1, d_weight, ldm, height, d_desc, boundary, tma.h_main[0]);
      if (t < cfg.timesteps - 1) {
        int32_t min_shifts_now;
        if (t == cfg.timesteps - 2)
          min_shifts_now = total_shifts_needed;
        else
          min_shifts_now = static_cast<int32_t>(std::ceil((t + 1) * log2_sf / 7.0));
        if (min_shifts_now < 0)
          min_shifts_now = 0;
        ozigpu::update_ozaki_desc<<<1, 32>>>(d_desc, min_shifts_now);
      }
      std::swap(d_slices_1, d_slices_2);
      tma.swap();
    }
    ev.record_stop();
    NVTX_POP();

    result.integration_ms.push_back(ev.elapsed_ms());
  }

  tma.free();

  // -- Mend (with warmup + repetition) --
  NVTX_PUSH("Mend");
  const int32_t nThreadsMend = std::min(static_cast<size_t>(RDC_THREADS), n_elems / 4);
  const int32_t nBlocksMend = ((n_elems / 4) + nThreadsMend - 1) / nThreadsMend;

  // Warmup
  if (cfg.warmup > 0) {
    ozigpu::mend<Oz><<<nBlocksMend, nThreadsMend>>>(d_data, d_slices_1, d_desc, n_elems);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Measured repeats (mend reads slices+desc without modifying them, so repeats are safe)
  for (int32_t r = 0; r < cfg.repeats; ++r) {
    ev_phase.record_start();
    ozigpu::mend<Oz><<<nBlocksMend, nThreadsMend>>>(d_data, d_slices_1, d_desc, n_elems);
    ev_phase.record_stop();
    result.mend_ms.push_back(ev_phase.elapsed_ms());
  }
  NVTX_POP();

  NVTX_POP(); // nvtx_label

  cudaFree(d_data);
  cudaFree(d_slices_1);
  cudaFree(d_slices_2);
  cudaFree(d_weight);
  cudaFree(d_desc);
  return result;
}

// M dispatch (switch 1..9)

template <size_t... Dims>
PhaseTimings dispatch_bench_direct(int m, const ozi::Stencil<CoeffStorageType, Dims...> &stencil,
                                   const std::vector<size_t> &shape, HighPrecType scaling_factor,
                                   CoeffStorageType a_coeff, int32_t extra_msb_shift, int32_t num_slices_total,
                                   const BenchConfig &cfg, ozigpu::L2CacheInvalidator &cache_inv) {
  // We need to construct stencil slices on the host
  auto do_bench = [&]<int32_t M_val>() -> PhaseTimings {
    auto ss = ozicpu::slice_stencil<ozicpu::OzakiFP_Int8<Real>, ozi::Stencil<CoeffStorageType, Dims...>, M_val>(
        stencil, a_coeff, extra_msb_shift, num_slices_total);
    return bench_ozaki_direct<M_val, Dims...>(ss, shape, num_slices_total, scaling_factor, cfg, cache_inv);
  };

  switch (m) {
  case 1:
    return do_bench.template operator()<1>();
  case 2:
    return do_bench.template operator()<2>();
  case 3:
    return do_bench.template operator()<3>();
  case 4:
    return do_bench.template operator()<4>();
  case 8:
    return do_bench.template operator()<8>();
  case 9:
    return do_bench.template operator()<9>();
  case 10:
    return do_bench.template operator()<10>();
  default:
    throw std::invalid_argument("M=" + std::to_string(m) + " is not a supported instantiation");
  }
}

template <size_t... Dims>
PhaseTimings dispatch_bench_conv(int m, const ozi::Stencil<CoeffStorageType, Dims...> &stencil,
                                 const std::vector<size_t> &shape, HighPrecType scaling_factor,
                                 CoeffStorageType a_coeff, int32_t extra_msb_shift, int32_t num_slices_total,
                                 const BenchConfig &cfg, ozigpu::L2CacheInvalidator &cache_inv) {
  if constexpr (sizeof...(Dims) == 2) {
    constexpr auto s_shape = ozi::Stencil<CoeffStorageType, Dims...>::shape;
    if constexpr (s_shape[0] == 7 && s_shape[1] == 7) {
      auto do_bench = [&]<int32_t M_val>() -> PhaseTimings {
        auto ss = ozicpu::slice_stencil<ozicpu::OzakiFP_Int8<Real>, ozi::Stencil<CoeffStorageType, Dims...>, M_val>(
            stencil, a_coeff, extra_msb_shift, num_slices_total);

        // Build weight matrix
        constexpr int32_t N_COEFFS = 7 * 7;
        int8_t flat_coeffs[M_val * N_COEFFS];
        for (int32_t si = 0; si < M_val; ++si)
          for (int32_t c = 0; c < N_COEFFS; ++c)
            flat_coeffs[si * N_COEFFS + c] = ss.slices[si].coeffs[c];

        constexpr size_t weight_size = M_val * 2 * ozigpu::convstencil::WT_SIZE;
        int8_t host_weight[weight_size];
        ozigpu::convstencil::build_param_matrix<M_val>(host_weight, flat_coeffs);

        // Build boundary gain data
        ozigpu::convstencil::BoundaryGainData boundary{};
        boundary.n_slices = ss.n_emulated_gain_slices;
        for (int32_t g = 0; g < boundary.n_slices; ++g)
          boundary.emulated_gain[g] = ss.emulated_gain[g];

        return bench_ozaki_conv<M_val>(ss, host_weight, boundary, shape, num_slices_total, scaling_factor, cfg,
                                       cache_inv);
      };

      switch (m) {
      case 1:
        return do_bench.template operator()<1>();
      case 2:
        return do_bench.template operator()<2>();
      case 3:
        return do_bench.template operator()<3>();
      case 4:
        return do_bench.template operator()<4>();
      case 8:
        return do_bench.template operator()<8>();
      case 9:
        return do_bench.template operator()<9>();
      case 10:
        return do_bench.template operator()<10>();
      default:
        throw std::invalid_argument("M=" + std::to_string(m) + " is not a supported instantiation");
      }
    }
  }
  throw std::invalid_argument("ConvStencil only supports 2D 7x7 stencils");
}

// Output formatting

static void print_header(const cudaDeviceProp &prop, const std::vector<size_t> &shape, const std::string &stencil_name,
                         int m, int slices, const BenchConfig &cfg) {
  std::cout << "================================================================\n";
  std::cout << "OzakiStencil Performance Benchmark\n";
  std::cout << "================================================================\n";
  std::cout << "Device:         " << prop.name << "\n";

  std::string shape_str;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      shape_str += " x ";
    shape_str += std::to_string(shape[i]);
  }
  std::cout << "Input shape:    " << shape_str << "\n";
  std::cout << "Stencil:        " << stencil_name << " (M=" << m << ")\n";
  std::cout << "Data slices:    " << slices << "\n";
  std::cout << "Timesteps:      " << cfg.timesteps << "\n";
  std::cout << "Warmup:         " << cfg.warmup << "\n";
  std::cout << "Repeats:        " << cfg.repeats << "\n";
  std::cout << "----------------------------------------------------------------\n";
  std::cout << std::left << std::setw(22) << "Algorithm"
            << std::right << std::setw(10) << "Min (ms)"
            << std::setw(10) << "Med (ms)"
            << std::setw(10) << "Max (ms)"
            << std::setw(9) << "GStPt/s" << "\n";
  std::cout << "----------------------------------------------------------------\n";
}

static void print_naive_row(const char *name, const PhaseTimings &t, const std::vector<size_t> &shape,
                            int32_t timesteps) {
  float mn = stat_min(t.integration_ms);
  float md = stat_median(t.integration_ms);
  float mx = stat_max(t.integration_ms);
  double gstpts = compute_gstpts(shape, timesteps, mn);
  std::cout << std::left << std::setw(22) << name
            << std::fixed << std::setprecision(2)
            << std::right << std::setw(10) << mn
            << std::setw(10) << md
            << std::setw(10) << mx
            << std::setw(9) << gstpts << "\n";
}

static void print_ozaki_row(const char *name, const PhaseTimings &t, const std::vector<size_t> &shape,
                            int32_t timesteps) {
  float dr_min = stat_min(t.dyn_range_ms), dr_med = stat_median(t.dyn_range_ms), dr_max = stat_max(t.dyn_range_ms);
  float sl_min = stat_min(t.slice_ms), sl_med = stat_median(t.slice_ms), sl_max = stat_max(t.slice_ms);
  float int_min = stat_min(t.integration_ms), int_med = stat_median(t.integration_ms),
        int_max = stat_max(t.integration_ms);
  float mend_min = stat_min(t.mend_ms), mend_med = stat_median(t.mend_ms), mend_max = stat_max(t.mend_ms);

  float total_min = dr_min + sl_min + int_min + mend_min;
  float total_med = dr_med + sl_med + int_med + mend_med;
  float total_max = dr_max + sl_max + int_max + mend_max;
  double gstpts = compute_gstpts(shape, timesteps, total_min);
  std::cout << std::left << std::setw(22) << name
            << std::fixed << std::setprecision(2)
            << std::right << std::setw(10) << total_min
            << std::setw(10) << total_med
            << std::setw(10) << total_max
            << std::setw(9) << gstpts << "\n";
  std::cout << "  " << std::left << std::setw(20) << "DynRange"
            << std::fixed << std::setprecision(2)
            << std::right << std::setw(10) << dr_min
            << std::setw(10) << dr_med
            << std::setw(10) << dr_max << "\n";
  std::cout << "  " << std::left << std::setw(20) << "Slice"
            << std::fixed << std::setprecision(2)
            << std::right << std::setw(10) << sl_min
            << std::setw(10) << sl_med
            << std::setw(10) << sl_max << "\n";

  double int_gstpts = compute_gstpts(shape, timesteps, int_min);
  std::cout << "  " << std::left << std::setw(20) << "Integration"
            << std::fixed << std::setprecision(2)
            << std::right << std::setw(10) << int_min
            << std::setw(10) << int_med
            << std::setw(10) << int_max
            << std::setw(9) << int_gstpts << "\n";
  std::cout << "  " << std::left << std::setw(20) << "Mend"
            << std::fixed << std::setprecision(2)
            << std::right << std::setw(10) << mend_min
            << std::setw(10) << mend_med
            << std::setw(10) << mend_max << "\n";
}

static void print_skipped_row(const char *name, const char *reason) {
  std::cout << std::left << std::setw(22) << name << " [skipped: " << reason << "]\n";
}

static void print_footer() { std::cout << "================================================================\n"; }

// Compute actual data slice count from dynamic range

static int32_t compute_actual_slices(const std::vector<size_t> &shape, const BenchConfig &cfg) {
  if (cfg.slices_override > 0)
    return cfg.slices_override;

  size_t n_elems = 1;
  for (auto s : shape)
    n_elems *= s;

  // Generate random data with the same seed and distribution as bench functions
  std::vector<Real> h_data(n_elems);
  std::mt19937_64 rng(cfg.seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (size_t i = 0; i < n_elems; ++i)
    h_data[i] = dist(rng);

  Real *d_data = nullptr;
  ozigpu::OzakiDescriptor<Oz> *d_desc = nullptr;
  ozigpu::OzakiDescriptor<Oz> h_desc;
  CUDA_CHECK(cudaMalloc(&d_data, n_elems * sizeof(Real)));
  CUDA_CHECK(cudaMalloc(&d_desc, sizeof(h_desc)));
  CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n_elems * sizeof(Real), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_desc, &h_desc, sizeof(h_desc), cudaMemcpyHostToDevice));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  const int32_t nBlocks = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / RDC_THREADS;

  ozigpu::dyn_range<Oz><<<nBlocks, RDC_THREADS>>>(d_desc, d_data, n_elems);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(&h_desc, d_desc, sizeof(h_desc), cudaMemcpyDeviceToHost));
  h_desc.e_max += cfg.extra_msb;
  h_desc.e_min -= cfg.extra_lsb;

  int32_t actual = 2 + (h_desc.e_max - h_desc.e_min) / (CHAR_BIT * sizeof(int8_t) - 1);

  cudaFree(d_data);
  cudaFree(d_desc);
  return actual;
}

// Shape dispatch + run all benchmarks for a single stencil

static void run_benchmarks(const StencilSpec &spec, const BenchConfig &cfg, int algo_flags, int device_cc) {
  auto params = ozi::compute_stencil_params(spec);
  auto variant = ozi::make_stencil(spec);

  // Compute the actual data slice count from the input's dynamic range.
  // This must happen before stencil slice construction so that emulated_gain
  // (boundary correction) is encoded with the correct number of base-128 digits.
  int32_t actual_slices = compute_actual_slices(cfg.shape, cfg);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  print_header(prop, cfg.shape, spec.name, params.m, actual_slices, cfg);

  ozigpu::L2CacheInvalidator cache_inv;

  std::visit(
      [&](const auto &stencil) {
        using StencilType = std::decay_t<decltype(stencil)>;
        constexpr size_t ndims = StencilType::dims;

        // Validate dimensionality
        if (ndims != cfg.shape.size()) {
          throw std::invalid_argument(
              "Stencil '" + spec.name + "' has " + std::to_string(ndims) +
              " dims but shape has " + std::to_string(cfg.shape.size()) + " dims");
        }

        // -- Naive FP32 --
        if (has_algo(algo_flags, AlgoFlag::NaiveFP32)) {
          auto t = bench_naive<float>(stencil, cfg.shape, cfg, cache_inv, "Naive FP32");
          print_naive_row("Naive FP32", t, cfg.shape, cfg.timesteps);
        }

        // -- Naive FP64 --
        if (has_algo(algo_flags, AlgoFlag::NaiveFP64)) {
          auto t = bench_naive<double>(stencil, cfg.shape, cfg, cache_inv, "Naive FP64");
          print_naive_row("Naive FP64", t, cfg.shape, cfg.timesteps);
        }

        // -- Naive FP128 --
        if (has_algo(algo_flags, AlgoFlag::NaiveFP128)) {
#if __CUDA_ARCH_LIST__ >= 1200 || !defined(__CUDA_ARCH_LIST__)
          if (device_cc >= 1200) {
            auto t = bench_naive<__float128>(stencil, cfg.shape, cfg, cache_inv, "Naive FP128");
            std::string label = "Naive FP128 [sm_" + std::to_string(device_cc / 10) + "]";
            print_naive_row(label.c_str(), t, cfg.shape, cfg.timesteps);
          } else {
            std::string reason = "device sm_" + std::to_string(device_cc / 10) + " < sm_120";
            print_skipped_row("Naive FP128", reason.c_str());
          }
#else
          std::string reason = "device sm_" + std::to_string(device_cc / 10) + " < sm_120";
          print_skipped_row("Naive FP128", reason.c_str());
#endif
        }

        // -- Ozaki Direct --
        if (has_algo(algo_flags, AlgoFlag::Direct)) {
          auto t = dispatch_bench_direct(params.m, stencil, cfg.shape, params.scaling_factor, params.a_coeff,
                                         params.extra_msb_shift, actual_slices, cfg, cache_inv);
          print_ozaki_row("Ozaki Direct", t, cfg.shape, cfg.timesteps);
        }

        // -- Ozaki ConvStencil --
        if (has_algo(algo_flags, AlgoFlag::ConvStencil)) {
          if constexpr (ndims == 2) {
            constexpr auto s_shape = StencilType::shape;
            if constexpr (s_shape[0] == 7 && s_shape[1] == 7) {
              auto t = dispatch_bench_conv(params.m, stencil, cfg.shape, params.scaling_factor, params.a_coeff,
                                           params.extra_msb_shift, actual_slices, cfg, cache_inv);
              print_ozaki_row("Ozaki ConvStencil", t, cfg.shape, cfg.timesteps);
            } else {
              print_skipped_row("Ozaki ConvStencil", "requires 2D 7x7 stencil");
            }
          } else {
            print_skipped_row("Ozaki ConvStencil", "requires 2D 7x7 stencil");
          }
        }
      },
      variant);

  print_footer();
}

// CLI parsing and main

static std::vector<size_t> parse_shape(const std::string &s) {
  std::vector<size_t> shape;
  std::istringstream ss(s);
  std::string dim;
  while (std::getline(ss, dim, 'x'))
    shape.push_back(std::stoull(dim));
  if (shape.empty())
    throw std::invalid_argument("Empty shape");
  return shape;
}

static dim3 parse_block_size(const std::string &s) {
  std::vector<unsigned> dims;
  std::istringstream ss(s);
  std::string token;
  while (std::getline(ss, token, 'x'))
    dims.push_back(static_cast<unsigned>(std::stoul(token)));
  if (dims.empty() || dims.size() > 3)
    throw std::invalid_argument("Block size must have 1-3 dimensions");
  return dim3(dims[0], dims.size() > 1 ? dims[1] : 1, dims.size() > 2 ? dims[2] : 1);
}

static void validate_block_size(const dim3 &block, const cudaDeviceProp &prop) {
  uint32_t total = block.x * block.y * block.z;
  if (total > static_cast<uint32_t>(prop.maxThreadsPerBlock))
    throw std::invalid_argument("Block size " + std::to_string(block.x) + "x" + std::to_string(block.y) + "x" +
                                std::to_string(block.z) + " = " + std::to_string(total) +
                                " threads exceeds device limit of " + std::to_string(prop.maxThreadsPerBlock));
  if (total % WARP_SIZE != 0)
    throw std::invalid_argument("Block size " + std::to_string(block.x) + "x" + std::to_string(block.y) + "x" +
                                std::to_string(block.z) + " = " + std::to_string(total) +
                                " threads is not a multiple of WARP_SIZE (" + std::to_string(WARP_SIZE) + ")");
}

int main(int argc, char **argv) {
  po::options_description desc("bench-stencil options");
  // clang-format off
  desc.add_options()
    ("help,h", "Show help")
    ("shape,S", po::value<std::string>()->required(), "Input grid shape (e.g., 1024x1024)")
    ("stencil", po::value<std::string>(), "Inline stencil spec: \"7x7:c0,c1,...\"")("stencil-file", po::value<std::string>(), "Stencil .stc file path")
    ("all-stencils", po::bool_switch(), "Run all default stencils")
    ("timesteps,T", po::value<int32_t>()->default_value(100), "Integration timesteps")
    ("warmup,W", po::value<int32_t>()->default_value(10), "Warmup timesteps")
    ("repeats,R", po::value<int32_t>()->default_value(3), "Measurement repeats")
    ("slices,s", po::value<int32_t>()->default_value(0), "Override data slice count")("extra-msb", po::value<int32_t>()->default_value(0), "Extra MSB headroom")
    ("extra-lsb", po::value<int32_t>()->default_value(0), "Extra LSB headroom")
    ("algorithm,A", po::value<std::string>()->default_value("all"), "Algorithms: naive-fp32,naive-fp64,naive-fp128,naive,direct,convstencil,all")
    ("seed", po::value<uint64_t>()->default_value(42), "Random seed")
    ("block-size,B", po::value<std::string>(), "Thread block dimensions for naive kernels (e.g., 256, 32x8, 8x8x4)")
    ("input,i", po::value<std::string>(), "Input tensor file")
    ("no-flush-l2", po::bool_switch(), "Skip L2 cache invalidation")
    ("ncu-mode", po::bool_switch(), "NCU capture mode (warmup=0, timesteps=1, repeats=1)");
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
    std::cerr << "Error: " << e.what() << "\n" << desc << "\n";
    return 1;
  }

  // Build config
  BenchConfig cfg;
  cfg.shape = parse_shape(vm["shape"].as<std::string>());
  cfg.timesteps = vm["timesteps"].as<int32_t>();
  cfg.warmup = vm["warmup"].as<int32_t>();
  cfg.repeats = vm["repeats"].as<int32_t>();
  cfg.slices_override = vm["slices"].as<int32_t>();
  cfg.extra_msb = vm["extra-msb"].as<int32_t>();
  cfg.extra_lsb = vm["extra-lsb"].as<int32_t>();
  cfg.seed = vm["seed"].as<uint64_t>();
  cfg.flush_l2 = !vm["no-flush-l2"].as<bool>();
  cfg.ncu_mode = vm["ncu-mode"].as<bool>();
  if (vm.count("block-size"))
    cfg.block_size = parse_block_size(vm["block-size"].as<std::string>());

  if (cfg.ncu_mode) {
    cfg.warmup = 0;
    cfg.timesteps = 1;
    cfg.repeats = 1;
  }

  // Get device compute capability
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int device_cc = prop.major * 100 + prop.minor * 10;

  if (cfg.block_size.x > 0)
    validate_block_size(cfg.block_size, prop);

  int algo_flags = parse_algos(vm["algorithm"].as<std::string>(), device_cc);

  // Collect stencil specs
  std::vector<StencilSpec> specs;
  if (vm["all-stencils"].as<bool>()) {
    specs = ozi::default_stencils();
  } else if (vm.count("stencil-file")) {
    specs.push_back(read_stencil_file(vm["stencil-file"].as<std::string>()));
  } else if (vm.count("stencil")) {
    specs.push_back(parse_stencil_arg(vm["stencil"].as<std::string>()));
  } else {
    std::cerr << "Error: must specify --stencil, --stencil-file, or --all-stencils\n";
    return 1;
  }

  NVTX_PUSH("bench-stencil");
  for (const auto &spec : specs) {
    NVTX_PUSH(spec.name.c_str());
    try {
      run_benchmarks(spec, cfg, algo_flags, device_cc);
    } catch (const std::exception &e) {
      std::cerr << "Error benchmarking '" << spec.name << "': " << e.what() << "\n";
    }
    NVTX_POP();
  }
  NVTX_POP();

  return 0;
}
