/**
 * GPU stencil kernel test suite.
 *
 * Test suites:
 *   1. NaiveGPUStencil  - FP64 shared-memory stencil (ozi::stencil)
 *   2. OzakiStencilV1   - Ozaki INT8 scalar stencil (ozigpu::stencil_v1)
 *   3. ConvStencilWMMA  - Ozaki INT8 tensor-core stencil (ozigpu::cvstencil)
 *   4. BoundaryTests    - Boundary cell correctness across kernels
 *   6. M > 1 Stencil Tests (multi-slice precision)
 *   7. Overflow/Shift Mechanism Tests
 */

#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <ozaki/core/stencil.hpp>
#include <ozaki/cpu/ozi.hpp>
#include <ozaki/gpu/gpu_checks.cuh>
#include <ozaki/gpu/ozi.cuh>
#include <ozaki/gpu/ozi_kernels.cuh>
#include <ozaki/gpu/ozi_stencil_conv.cuh>
#include <ozaki/gpu/ozi_stencil_plain.cuh>
#include <ozaki/gpu/stencil_gpu.cuh>
#include <ozaki/utils/stencil_registry.hpp>

#include "../test_utils.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#ifdef OZAKI_HAVE_FLOAT128
#include <quadmath.h>
#endif

// ============================================================
// Type aliases
// ============================================================
using Real = double;
#if defined(OZAKI_CUDA_ARCH) && OZAKI_CUDA_ARCH >= 100
using AccType = __float128;
#else
using AccType = double;
#endif
using OzPair = ozigpu::OzakiFP_Int8<Real>;

// ============================================================
// Stencil parameter computation helper
// ============================================================
struct OzakiSetup {
  HighPrecType a_coeff;
  int32_t m;
  HighPrecType scaling_factor;
  int32_t extra_msb_shift;
};

template <size_t... Dims> static OzakiSetup compute_ozaki_setup(const ozi::Stencil<Real, Dims...> &s) {
  StencilSpec spec;
  spec.name = "test";
  for (auto d : s.get_shape())
    spec.shape.push_back(static_cast<int>(d));
  for (auto o : s.offsets)
    spec.offsets.push_back(static_cast<int>(o));
  for (size_t i = 0; i < s.size; ++i)
    spec.coeffs.push_back(static_cast<CoeffStorageType>(s.coeffs[i]));

  auto params = ozi::compute_stencil_params(spec);
  return {params.a_coeff, params.m, params.scaling_factor, params.extra_msb_shift};
}

// ============================================================
// GPU Test Fixture
// ============================================================
class StencilGPUTest : public ::testing::Test {
protected:
  static constexpr uint64_t SEED = 0xABCDEF01;
  static constexpr int32_t RDC_THREADS = RDC_THREADS_PER_BLOCK;

  Real *d_data = nullptr;
  int8_t *d_slices_1 = nullptr, *d_slices_2 = nullptr;
  ozigpu::OzakiDescriptor<OzPair> *d_desc = nullptr;
  ozigpu::OzakiDescriptor<OzPair> h_desc;
  AccType *d_res = nullptr;

  size_t n_elems = 0;
  int32_t num_slices_total = 0;

  void SetUp() override {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
      GTEST_SKIP() << "No CUDA device found";
    cudaSetDevice(0);
  }

  void TearDown() override {
    if (d_data)     { cudaFree(d_data);     d_data = nullptr; }
    if (d_slices_1) { cudaFree(d_slices_1); d_slices_1 = nullptr; }
    if (d_slices_2) { cudaFree(d_slices_2); d_slices_2 = nullptr; }
    if (d_desc)     { cudaFree(d_desc);     d_desc = nullptr; }
    if (d_res)      { cudaFree(d_res);      d_res = nullptr; }
  }

  // --- Data generation ---

  std::vector<Real> generate_random(size_t size, double mean = 0.0, double stddev = 1.0) {
    std::vector<Real> vec(size);
    std::mt19937 engine(SEED);
    std::normal_distribution<double> dist(mean, stddev);
    for (auto &v : vec)
      v = dist(engine);
    return vec;
  }

  // --- CUDA resource management ---

  void alloc_and_upload(const std::vector<Real> &input) {
    n_elems = input.size();
    ASSERT_EQ(n_elems % 4, 0u) << "n_elems must be divisible by 4 for vectorized kernels";

    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_data, n_elems * sizeof(Real)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_data, input.data(), n_elems * sizeof(Real), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_desc, sizeof(ozigpu::OzakiDescriptor<OzPair>)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_res, n_elems * sizeof(AccType)));
  }

  void compute_dynamic_range(int32_t extra_msb = 0, int32_t extra_lsb = 0) {
    h_desc = ozigpu::OzakiDescriptor<OzPair>();
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_desc, &h_desc, sizeof(h_desc), cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int32_t nBlocks = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / RDC_THREADS;

    ozigpu::dyn_range<<<nBlocks, RDC_THREADS>>>(d_desc, d_data, n_elems);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&h_desc, d_desc, sizeof(h_desc), cudaMemcpyDeviceToHost));

    h_desc.e_max += extra_msb;
    h_desc.e_min -= extra_lsb;
    num_slices_total = 2 + (h_desc.e_max - h_desc.e_min) / (CHAR_BIT * sizeof(int8_t) - 1);
    h_desc.n_slices_input = num_slices_total;

    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_desc, &h_desc, sizeof(h_desc), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_slices_1, num_slices_total * n_elems * sizeof(int8_t) + ozigpu::STENCIL_V1_ALLOC_PAD));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_slices_2, num_slices_total * n_elems * sizeof(int8_t) + ozigpu::STENCIL_V1_ALLOC_PAD));
  }

  void prepare_slices() {
    int32_t nThreads = std::min(static_cast<size_t>(RDC_THREADS), n_elems / 4);
    int32_t nBlocks = static_cast<int32_t>(((n_elems / 4) + nThreads - 1) / nThreads);
    ozigpu::slice<OzPair><<<nBlocks, nThreads>>>(d_slices_1, d_data, d_desc, n_elems);
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(d_slices_2, d_slices_1, num_slices_total * n_elems * sizeof(int8_t), cudaMemcpyDeviceToDevice));
  }

  // --- Stencil_v1 pipeline helpers ---

  template <int32_t M, size_t... Dims>
  void configure_descriptor(const ozi::StencilSlices<ozi::Stencil<int8_t, Dims...>, M> &ss) {
    double slice0_l1 = ss.slices[0].template l1_norm<double>();
    int32_t l1_int = std::max(1, static_cast<int32_t>(slice0_l1));

    h_desc.shift_required = 0;
    h_desc.carry_slice_index = 0;
    h_desc.overflow_danger_threshold = (std::numeric_limits<int8_t>::max() / l1_int) - 1;
    h_desc.just_shifted = 1;
    h_desc.n_shifts = 0;

    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_desc, &h_desc, sizeof(h_desc), cudaMemcpyHostToDevice));
  }

  template <int32_t M, size_t... Dims>
  void run_stencil_v1_loop(const ozi::StencilSlices<ozi::Stencil<int8_t, Dims...>, M> &ss,
                           const std::array<size_t, sizeof...(Dims)> &shape, int32_t timesteps,
                           HighPrecType scaling_factor) {
    constexpr size_t ndims = sizeof...(Dims);
    dim3 block, grid;

    // Grid config: i->fastest dim (last), j->second, k->slowest (first)
    if constexpr (ndims == 1) {
      block = dim3(256);
      grid = dim3((shape[0] + 255) / 256);
    } else if constexpr (ndims == 2) {
      block = dim3(32, 16);
      grid = dim3((shape[1] + 31) / 32, (shape[0] + 15) / 16);
    } else {
      block = dim3(8, 8, 4);
      grid = dim3((shape[2] + 7) / 8, (shape[1] + 7) / 8, (shape[0] + 3) / 4);
    }

    typename ozi::Stencil<int8_t, Dims...>::shape_type input_shape{};
    for (size_t d = 0; d < ndims; ++d)
      input_shape[d] = shape[d];

    double log2_sf =
        (scaling_factor > static_cast<HighPrecType>(1)) ? std::log2(static_cast<double>(scaling_factor)) : 0.0;
    int32_t total_shifts_needed = static_cast<int32_t>(std::ceil(timesteps * log2_sf / 7.0));

    // Copy stencil slices to constant memory before kernel launch
    ASSERT_EQ(cudaSuccess, cudaMemcpyToSymbol(ozigpu::stencil_v1_cmem, &ss, sizeof(ss)));
    ozigpu::upload_slice_phys_lut(num_slices_total);

    for (int32_t t = 0; t < timesteps; ++t) {
      ozigpu::stencil_v1<OzPair, M, Dims...><<<grid, block>>>(d_slices_2, d_slices_1, input_shape, d_desc);

      if (t < timesteps - 1) {
        int32_t min_shifts_now;
        if (t == timesteps - 2)
          min_shifts_now = total_shifts_needed;
        else
          min_shifts_now = static_cast<int32_t>(std::ceil((t + 1) * log2_sf / 7.0));
        if (min_shifts_now < 0)
          min_shifts_now = 0;
        ozigpu::update_ozaki_desc<<<1, 32>>>(d_desc, min_shifts_now);
      }
      std::swap(d_slices_1, d_slices_2);
    }
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  }

  std::vector<Real> mend_scale_download(int32_t timesteps, HighPrecType scaling_factor) {
    int32_t nTMend = static_cast<int32_t>(std::min(static_cast<size_t>(RDC_THREADS), n_elems / 4));
    int32_t nBMend = static_cast<int32_t>(((n_elems / 4) + nTMend - 1) / nTMend);
    int32_t nTScale = static_cast<int32_t>(std::min(static_cast<size_t>(RDC_THREADS), n_elems));
    int32_t nBScale = static_cast<int32_t>((n_elems + nTScale - 1) / nTScale);

    ozigpu::mend<OzPair, AccType><<<nBMend, nTMend>>>(d_res, d_slices_1, d_desc, n_elems);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(&h_desc, d_desc, sizeof(h_desc), cudaMemcpyDeviceToHost));

#if defined(OZAKI_CUDA_ARCH) && OZAKI_CUDA_ARCH >= 100
    __float128 factor =
        ozigpu::total_integration_scale_factor_quad<OzPair>(scaling_factor, timesteps, h_desc.n_shifts);
    ozigpu::scale<AccType, __float128><<<nBScale, nTScale>>>(d_res, factor, n_elems);
#else
    double factor = static_cast<double>(
        ozigpu::total_integration_scale_factor_quad<OzPair>(scaling_factor, timesteps, h_desc.n_shifts));
    ozigpu::scale<AccType, double><<<nBScale, nTScale>>>(d_res, factor, n_elems);
#endif
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    std::vector<AccType> result_q(n_elems);
    EXPECT_EQ(cudaSuccess, cudaMemcpy(result_q.data(), d_res, n_elems * sizeof(AccType), cudaMemcpyDeviceToHost));

    std::vector<Real> result(n_elems);
    for (size_t i = 0; i < n_elems; ++i)
      result[i] = static_cast<Real>(result_q[i]);
    return result;
  }

  // --- High-level pipeline: run full Ozaki stencil_v1 ---

  template <int32_t M, size_t... Dims>
  std::vector<Real> run_full_ozaki_v1(const ozi::Stencil<Real, Dims...> &stencil,
                                      const std::vector<Real> &input,
                                      const std::array<size_t, sizeof...(Dims)> &shape, int32_t timesteps,
                                      HighPrecType a_coeff, HighPrecType scaling_factor,
                                      int32_t extra_msb_shift = 0) {
    alloc_and_upload(input);
    compute_dynamic_range();

    auto ss = ozicpu::slice_stencil<ozicpu::OzakiFP_Int8<Real>, ozi::Stencil<Real, Dims...>, M>(
        stencil, a_coeff, extra_msb_shift, num_slices_total);

    configure_descriptor(ss);
    prepare_slices();
    run_stencil_v1_loop(ss, shape, timesteps, scaling_factor);
    return mend_scale_download(timesteps, scaling_factor);
  }

  // --- High-level pipeline: run full ConvStencil ---

  template <int32_t M>
  std::vector<Real> run_full_cvstencil(const ozi::Stencil<Real, 7, 7> &stencil, const std::vector<Real> &input,
                                       const std::array<size_t, 2> &shape, int32_t timesteps,
                                       HighPrecType a_coeff, HighPrecType scaling_factor,
                                       int32_t extra_msb_shift = 0) {
    alloc_and_upload(input);
    compute_dynamic_range();

    auto ss = ozicpu::slice_stencil<ozicpu::OzakiFP_Int8<Real>, ozi::Stencil<Real, 7, 7>, M>(
        stencil, a_coeff, extra_msb_shift, num_slices_total);

    configure_descriptor(ss);
    prepare_slices();

    // Build weight matrix
    constexpr int32_t N_COEFFS = 7 * 7;
    int8_t flat_coeffs[M * N_COEFFS];
    for (int32_t si = 0; si < M; ++si)
      for (int32_t c = 0; c < N_COEFFS; ++c)
        flat_coeffs[si * N_COEFFS + c] = ss.slices[si].coeffs[c];

    constexpr size_t weight_size = M * 2 * ozigpu::convstencil::WT_SIZE;
    int8_t host_weight[weight_size];
    ozigpu::convstencil::build_param_matrix<M>(host_weight, flat_coeffs);

    int8_t *d_weight = nullptr;
    EXPECT_EQ(cudaSuccess, cudaMalloc(&d_weight, weight_size));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(d_weight, host_weight, weight_size, cudaMemcpyHostToDevice));

    // Build boundary gain data
    ozigpu::convstencil::BoundaryGainData boundary{};
    boundary.n_slices = ss.n_emulated_gain_slices;
    for (int32_t g = 0; g < boundary.n_slices; ++g)
      boundary.emulated_gain[g] = ss.emulated_gain[g];

    // Grid/block
    const size_t height = shape[0];
    const size_t ldm = shape[1];
    constexpr int32_t BLOCK_THREADS = ozigpu::convstencil::NUM_WARPS * WARP_SIZE;
    dim3 block(BLOCK_THREADS);
    dim3 grid((height + ozigpu::convstencil::TILE_SIZE_ROW - 1) / ozigpu::convstencil::TILE_SIZE_ROW,
              (ldm + ozigpu::convstencil::TILE_SIZE_COL - 1) / ozigpu::convstencil::TILE_SIZE_COL);

    // Shift schedule
    double log2_sf =
        (scaling_factor > static_cast<HighPrecType>(1)) ? std::log2(static_cast<double>(scaling_factor)) : 0.0;
    int32_t total_shifts_needed = static_cast<int32_t>(std::ceil(timesteps * log2_sf / 7.0));

    ozigpu::upload_slice_phys_lut(num_slices_total);

    ozigpu::convstencil::TmaDescriptors tma;
    size_t total_height = static_cast<size_t>(num_slices_total) * height;
    bool tma_ok = ozigpu::convstencil::create_tma_descriptors(tma, d_slices_1, d_slices_2, total_height, ldm);
    if (!tma_ok)
      throw std::runtime_error("cvstencil requires ldm >= 128 and ldm % 16 == 0; got ldm=" + std::to_string(ldm));

    for (int32_t t = 0; t < timesteps; ++t) {
      ozigpu::cvstencil<OzPair, M><<<grid, block>>>(d_slices_2, d_slices_1, d_weight, ldm, height, d_desc, boundary, tma.d_main[0], tma.d_extra[0]);

      if (t < timesteps - 1) {
        int32_t min_shifts_now;
        if (t == timesteps - 2)
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
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

    tma.free();
    cudaFree(d_weight);
    return mend_scale_download(timesteps, scaling_factor);
  }
};

// ============================================================
// Test Suite 1: NaiveGPUStencil
// ============================================================

TEST_F(StencilGPUTest, Naive_Identity1D) {
  constexpr size_t N = 1024;
  auto input = generate_random(N);

  ozi::Stencil<Real, 3> identity({0.0, 1.0, 0.0});

  // Upload stencil to constant memory
  ASSERT_EQ(cudaSuccess, cudaMemcpyToSymbol(ozi::stencil_cmem, &identity, sizeof(identity)));

  // Allocate device buffers
  Real *d_in = nullptr, *d_out = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_in, N * sizeof(Real)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_out, N * sizeof(Real)));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_in, input.data(), N * sizeof(Real), cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemset(d_out, 0, N * sizeof(Real)));

  dim3 block(256);
  dim3 grid((N + 255) / 256);

  std::array<size_t, 1> shape = {N};
  ozi::stencil_fp64<3><<<grid, block>>>(d_out, d_in, shape);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  std::vector<Real> result(N);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(result.data(), d_out, N * sizeof(Real), cudaMemcpyDeviceToHost));

  // Interior cells should be bit-exact copy
  for (size_t i = identity.offsets[0]; i < N - identity.offsets[0]; ++i)
    EXPECT_EQ(result[i], input[i]) << "Mismatch at " << i;

  cudaFree(d_in);
  cudaFree(d_out);
}

TEST_F(StencilGPUTest, Naive_SecondOrderFD1D) {
  constexpr size_t N = 1024;
  auto input = generate_random(N);

  ozi::Stencil<Real, 3> d2x({1.0, -2.0, 1.0});
  ASSERT_EQ(cudaSuccess, cudaMemcpyToSymbol(ozi::stencil_cmem, &d2x, sizeof(d2x)));

  Real *d_in = nullptr, *d_out = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_in, N * sizeof(Real)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_out, N * sizeof(Real)));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_in, input.data(), N * sizeof(Real), cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemset(d_out, 0, N * sizeof(Real)));

  dim3 block(256);
  dim3 grid((N + 255) / 256);

  std::array<size_t, 1> shape = {N};
  ozi::stencil_fp64<3><<<grid, block>>>(d_out, d_in, shape);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  std::vector<Real> gpu_result(N);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(gpu_result.data(), d_out, N * sizeof(Real), cudaMemcpyDeviceToHost));

  // CPU reference
  std::vector<Real> cpu_result(N, 0.0);
  d2x.template apply_to<Real>(cpu_result, input, shape);

  for (size_t i = d2x.offsets[0]; i < N - d2x.offsets[0]; ++i)
    EXPECT_DOUBLE_EQ(gpu_result[i], cpu_result[i]) << "Mismatch at " << i;

  cudaFree(d_in);
  cudaFree(d_out);
}

TEST_F(StencilGPUTest, Naive_Laplace2D) {
  constexpr size_t H = 64, W = 64;
  auto input = generate_random(H * W);

  ozi::Stencil<Real, 3, 3> laplacian({0, 1, 0, 1, -4, 1, 0, 1, 0});
  ASSERT_EQ(cudaSuccess, cudaMemcpyToSymbol(ozi::stencil_cmem, &laplacian, sizeof(laplacian)));

  Real *d_in = nullptr, *d_out = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_in, H * W * sizeof(Real)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_out, H * W * sizeof(Real)));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_in, input.data(), H * W * sizeof(Real), cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemset(d_out, 0, H * W * sizeof(Real)));

  dim3 block(32, 16);
  dim3 grid((W + 31) / 32, (H + 15) / 16);

  std::array<size_t, 2> shape = {H, W};
  ozi::stencil_fp64<3, 3><<<grid, block>>>(d_out, d_in, shape);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  std::vector<Real> gpu_result(H * W);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(gpu_result.data(), d_out, H * W * sizeof(Real), cudaMemcpyDeviceToHost));

  std::vector<Real> cpu_result(H * W, 0.0);
  laplacian.template apply_to<Real>(cpu_result, input, shape);

  double rmse =
      test_utils::interior_relative_rmse<Real, 2>(gpu_result, cpu_result, {H, W}, laplacian.offsets);
  EXPECT_LT(rmse, 1e-14) << "GPU vs CPU RMSE too high for 2D Laplacian";

  cudaFree(d_in);
  cudaFree(d_out);
}

TEST_F(StencilGPUTest, Naive_Box3D) {
  constexpr size_t D = 32, H = 32, W = 32;
  auto input = generate_random(D * H * W);

  ozi::Stencil<Real, 3, 3, 3> box;
  for (auto &c : box.coeffs)
    c = 1.0 / 27.0;
  ASSERT_EQ(cudaSuccess, cudaMemcpyToSymbol(ozi::stencil_cmem, &box, sizeof(box)));

  Real *d_in = nullptr, *d_out = nullptr;
  size_t total = D * H * W;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_in, total * sizeof(Real)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_out, total * sizeof(Real)));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_in, input.data(), total * sizeof(Real), cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemset(d_out, 0, total * sizeof(Real)));

  dim3 block(8, 8, 4);
  dim3 grid((W + 7) / 8, (H + 7) / 8, (D + 3) / 4);

  std::array<size_t, 3> shape = {D, H, W};
  ozi::stencil_fp64<3, 3, 3><<<grid, block>>>(d_out, d_in, shape);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  std::vector<Real> gpu_result(total);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(gpu_result.data(), d_out, total * sizeof(Real), cudaMemcpyDeviceToHost));

  std::vector<Real> cpu_result(total, 0.0);
  box.template apply_to<Real>(cpu_result, input, shape);

  double rmse = test_utils::interior_relative_rmse<Real, 3>(gpu_result, cpu_result, {D, H, W}, box.offsets);
  EXPECT_LT(rmse, 1e-14) << "GPU vs CPU RMSE too high for 3D Box";

  cudaFree(d_in);
  cudaFree(d_out);
}

TEST_F(StencilGPUTest, Naive_PolynomialExact2D) {
  constexpr size_t H = 64, W = 64;

  // u(x,y) = x^2 + y^2, Laplacian should give 4.0 for interior cells
  std::vector<Real> input(H * W);
  for (size_t r = 0; r < H; ++r)
    for (size_t c = 0; c < W; ++c)
      input[r * W + c] = static_cast<Real>(r * r + c * c);

  ozi::Stencil<Real, 3, 3> laplacian({0, 1, 0, 1, -4, 1, 0, 1, 0});
  ASSERT_EQ(cudaSuccess, cudaMemcpyToSymbol(ozi::stencil_cmem, &laplacian, sizeof(laplacian)));

  Real *d_in = nullptr, *d_out = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_in, H * W * sizeof(Real)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_out, H * W * sizeof(Real)));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_in, input.data(), H * W * sizeof(Real), cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemset(d_out, 0, H * W * sizeof(Real)));

  dim3 block(32, 16);
  dim3 grid((W + 31) / 32, (H + 15) / 16);

  std::array<size_t, 2> shape = {H, W};
  ozi::stencil_fp64<3, 3><<<grid, block>>>(d_out, d_in, shape);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  std::vector<Real> result(H * W);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(result.data(), d_out, H * W * sizeof(Real), cudaMemcpyDeviceToHost));

  // Interior cells should be exactly 4.0 (d2u/dx2 + d2u/dy2 = 2 + 2 = 4, grid spacing h=1)
  for (size_t r = 1; r < H - 1; ++r)
    for (size_t c = 1; c < W - 1; ++c)
      EXPECT_DOUBLE_EQ(result[r * W + c], 4.0) << "At (" << r << "," << c << ")";

  cudaFree(d_in);
  cudaFree(d_out);
}

// ============================================================
// Test Suite 2: OzakiStencilV1
// ============================================================

TEST_F(StencilGPUTest, OzakiV1_SingleStep_d2x_1D) {
  constexpr size_t N = 1024;
  auto input = generate_random(N);

  ozi::Stencil<Real, 3> d2x({1.0, -2.0, 1.0});
  auto setup = compute_ozaki_setup(d2x);
  ASSERT_EQ(setup.m, 1);

  auto result = run_full_ozaki_v1<1>(d2x, input, {N}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);
  auto gold = test_utils::compute_gold_standard(d2x, input, std::array<size_t, 1>{N});

  double rmse = test_utils::interior_relative_rmse<Real, 1>(result, gold, {N}, d2x.offsets);
  EXPECT_LT(rmse, 1e-12) << "1D d2x single-step RMSE too high: " << rmse;
}

TEST_F(StencilGPUTest, OzakiV1_SingleStepHeat2D) {
  constexpr size_t H = 64, W = 64;
  auto input = generate_random(H * W);

  ozi::Stencil<Real, 3, 3> heat({0, 3.0 / 16, 0, 3.0 / 16, 4.0 / 16, 3.0 / 16, 0, 3.0 / 16, 0});
  auto setup = compute_ozaki_setup(heat);
  ASSERT_EQ(setup.m, 1);

  auto result =
      run_full_ozaki_v1<1>(heat, input, {H, W}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);
  auto gold = test_utils::compute_gold_standard(heat, input, std::array<size_t, 2>{H, W});

  double rmse = test_utils::interior_relative_rmse<Real, 2>(result, gold, {H, W}, heat.offsets);
  EXPECT_LT(rmse, 1e-12) << "2D Heat single-step RMSE: " << rmse;
}

TEST_F(StencilGPUTest, OzakiV1_SingleStepAdvDiff2D) {
  constexpr size_t H = 128, W = 128;
  auto input = generate_random(H * W);

  ozi::Stencil<Real, 3, 3> advdiff(
      {0, 11.0 / 61, 0, 11.0 / 61, 15.0 / 61, 12.0 / 61, 0, 12.0 / 61, 0});
  auto setup = compute_ozaki_setup(advdiff);
  ASSERT_EQ(setup.m, 1);

  auto result =
      run_full_ozaki_v1<1>(advdiff, input, {H, W}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);
  auto gold = test_utils::compute_gold_standard(advdiff, input, std::array<size_t, 2>{H, W});

  double rmse = test_utils::interior_relative_rmse<Real, 2>(result, gold, {H, W}, advdiff.offsets);
  EXPECT_LT(rmse, 1e-12) << "2D AdvDiff single-step RMSE: " << rmse;
}

TEST_F(StencilGPUTest, OzakiV1_SingleStep3D) {
  constexpr size_t D = 32, H = 32, W = 32;
  auto input = generate_random(D * H * W);

  // Uniform 3x3x3 box: L1 = 1.0
  ozi::Stencil<Real, 3, 3, 3> box;
  for (auto &c : box.coeffs)
    c = 1.0 / 27.0;
  auto setup = compute_ozaki_setup(box);

  auto result = run_full_ozaki_v1<1>(box, input, {D, H, W}, 1, setup.a_coeff, setup.scaling_factor,
                                     setup.extra_msb_shift);
  auto gold = test_utils::compute_gold_standard(box, input, std::array<size_t, 3>{D, H, W});

  double rmse = test_utils::interior_relative_rmse<Real, 3>(result, gold, {D, H, W}, box.offsets);
  EXPECT_LT(rmse, 1e-12) << "3D Box single-step RMSE: " << rmse;
}

TEST_F(StencilGPUTest, OzakiV1_SingleStepNonAligned2D) {
  // Non-power-of-2 grid (must still be multiple of 4 for vectorized kernels)
  constexpr size_t H = 68, W = 36;
  static_assert((H * W) % 4 == 0);
  auto input = generate_random(H * W);

  ozi::Stencil<Real, 3, 3> heat({0, 3.0 / 16, 0, 3.0 / 16, 4.0 / 16, 3.0 / 16, 0, 3.0 / 16, 0});
  auto setup = compute_ozaki_setup(heat);

  auto result =
      run_full_ozaki_v1<1>(heat, input, {H, W}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);
  auto gold = test_utils::compute_gold_standard(heat, input, std::array<size_t, 2>{H, W});

  double rmse = test_utils::interior_relative_rmse<Real, 2>(result, gold, {H, W}, heat.offsets);
  EXPECT_LT(rmse, 1e-12) << "Non-aligned 2D RMSE: " << rmse;
}

// --- Multi-step tests ---

TEST_F(StencilGPUTest, OzakiV1_MultiStep100_Stable2D) {
  constexpr size_t H = 64, W = 64;
  constexpr int32_t T = 100;
  auto input = generate_random(H * W);

  // Stable stencil: Heat * 0.9 (L1 = 0.9 < 1, guaranteed no divergence)
  ozi::Stencil<Real, 3, 3> stable_heat(
      {0, 0.9 * 3.0 / 16, 0, 0.9 * 3.0 / 16, 0.9 * 4.0 / 16, 0.9 * 3.0 / 16, 0, 0.9 * 3.0 / 16, 0});
  auto setup = compute_ozaki_setup(stable_heat);

  auto result = run_full_ozaki_v1<1>(stable_heat, input, {H, W}, T, setup.a_coeff, setup.scaling_factor,
                                     setup.extra_msb_shift);
  auto gold = test_utils::compute_gold_standard_multistep(stable_heat, input, std::array<size_t, 2>{H, W}, T);

  double rmse = test_utils::interior_relative_rmse<Real, 2>(result, gold, {H, W}, stable_heat.offsets);
  EXPECT_LT(rmse, T * 5e-12) << "Multi-step (T=100) stable RMSE: " << rmse;
}

TEST_F(StencilGPUTest, OzakiV1_MultiStep10_1D) {
  constexpr size_t N = 512;
  constexpr int32_t T = 10;
  auto input = generate_random(N);

  // Averaging stencil: {0.25, 0.5, 0.25}, L1 = 1.0
  ozi::Stencil<Real, 3> avg({0.25, 0.5, 0.25});
  auto setup = compute_ozaki_setup(avg);

  auto result =
      run_full_ozaki_v1<1>(avg, input, {N}, T, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);
  auto gold = test_utils::compute_gold_standard_multistep(avg, input, std::array<size_t, 1>{N}, T);

  double rmse = test_utils::interior_relative_rmse<Real, 1>(result, gold, {N}, avg.offsets);
  EXPECT_LT(rmse, T * 5e-12) << "Multi-step (T=10) 1D RMSE: " << rmse;
}

// --- Edge cases ---

TEST_F(StencilGPUTest, OzakiV1_AllZeroInput) {
  constexpr size_t H = 64, W = 64;
  std::vector<Real> input(H * W, 0.0);

  ozi::Stencil<Real, 3, 3> heat({0, 3.0 / 16, 0, 3.0 / 16, 4.0 / 16, 3.0 / 16, 0, 3.0 / 16, 0});
  auto setup = compute_ozaki_setup(heat);

  auto result =
      run_full_ozaki_v1<1>(heat, input, {H, W}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);

  for (size_t i = 0; i < result.size(); ++i)
    EXPECT_EQ(result[i], 0.0) << "Non-zero output for zero input at " << i;
}

TEST_F(StencilGPUTest, OzakiV1_ConstantInput) {
  constexpr size_t H = 64, W = 64;
  constexpr Real C = 42.0;
  std::vector<Real> input(H * W, C);

  // Heat stencil with L1=1: should preserve constant
  ozi::Stencil<Real, 3, 3> heat({0, 3.0 / 16, 0, 3.0 / 16, 4.0 / 16, 3.0 / 16, 0, 3.0 / 16, 0});
  auto setup = compute_ozaki_setup(heat);

  auto result =
      run_full_ozaki_v1<1>(heat, input, {H, W}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);

  // Interior cells should be C (stencil coeffs sum to 1)
  for (size_t r = 1; r < H - 1; ++r)
    for (size_t c = 1; c < W - 1; ++c) {
      double err = std::abs(result[r * W + c] - C);
      EXPECT_LT(err, 1e-10) << "Constant preservation failed at (" << r << "," << c << "): got "
                             << result[r * W + c];
    }
}

TEST_F(StencilGPUTest, OzakiV1_CatastrophicCancellation2D) {
  constexpr size_t H = 64, W = 64;

  // Data with huge dynamic range: mix of large and small values
  std::vector<Real> input(H * W);
  std::mt19937 engine(SEED);
  std::uniform_int_distribution<int> sign_dist(0, 1);
  for (size_t i = 0; i < H * W; ++i) {
    double base = (i % 3 == 0) ? 1e15 : ((i % 3 == 1) ? -1e15 : 1.0);
    input[i] = base + (sign_dist(engine) ? 0.001 : -0.001);
  }

  ozi::Stencil<Real, 3, 3> heat({0, 3.0 / 16, 0, 3.0 / 16, 4.0 / 16, 3.0 / 16, 0, 3.0 / 16, 0});
  auto setup = compute_ozaki_setup(heat);

  auto result =
      run_full_ozaki_v1<1>(heat, input, {H, W}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);
  auto gold = test_utils::compute_gold_standard(heat, input, std::array<size_t, 2>{H, W});

  double rmse = test_utils::interior_relative_rmse<Real, 2>(result, gold, {H, W}, heat.offsets);
  EXPECT_LT(rmse, 1e-12) << "Catastrophic cancellation RMSE: " << rmse;
}

// ============================================================
// Test Suite 3: ConvStencilWMMA (7x7 stencils only)
// ============================================================

TEST_F(StencilGPUTest, CvStencil_SingleStepBox7x7) {
  constexpr size_t H = 128, W = 128;
  auto input = generate_random(H * W);

  // Uniform 7x7 box: all coefficients = 1/49
  ozi::Stencil<Real, 7, 7> box;
  for (auto &c : box.coeffs)
    c = 1.0 / 49.0;
  auto setup = compute_ozaki_setup(box);
  ASSERT_EQ(setup.m, 1);

  auto result =
      run_full_cvstencil<1>(box, input, {H, W}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);
  auto gold = test_utils::compute_gold_standard(box, input, std::array<size_t, 2>{H, W});

  double rmse = test_utils::interior_relative_rmse<Real, 2>(result, gold, {H, W}, box.offsets);
  EXPECT_LT(rmse, 1e-12) << "CvStencil box7x7 single-step RMSE: " << rmse;
}

TEST_F(StencilGPUTest, CvStencil_MatchesStencilV1) {
  constexpr size_t H = 128, W = 128;
  auto input = generate_random(H * W);

  ozi::Stencil<Real, 7, 7> box;
  for (auto &c : box.coeffs)
    c = 1.0 / 49.0;
  auto setup = compute_ozaki_setup(box);

  // Run both kernels on the same input
  auto result_cv =
      run_full_cvstencil<1>(box, input, {H, W}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);

  // Need fresh GPU state for stencil_v1
  TearDown();
  SetUp();

  auto result_v1 =
      run_full_ozaki_v1<1>(box, input, {H, W}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);

  double rmse = test_utils::interior_relative_rmse<Real, 2>(result_cv, result_v1, {H, W}, box.offsets);
  EXPECT_LT(rmse, 1e-14) << "CvStencil vs StencilV1 cross-kernel RMSE: " << rmse;
}

TEST_F(StencilGPUTest, CvStencil_MultiStep10) {
  constexpr size_t H = 128, W = 128;
  constexpr int32_t T = 10;
  auto input = generate_random(H * W);

  ozi::Stencil<Real, 7, 7> box;
  for (auto &c : box.coeffs)
    c = 1.0 / 49.0;
  auto setup = compute_ozaki_setup(box);

  auto result =
      run_full_cvstencil<1>(box, input, {H, W}, T, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);
  auto gold = test_utils::compute_gold_standard_multistep(box, input, std::array<size_t, 2>{H, W}, T);

  double rmse = test_utils::interior_relative_rmse<Real, 2>(result, gold, {H, W}, box.offsets);
  EXPECT_LT(rmse, T * 5e-12) << "CvStencil multi-step (T=10) RMSE: " << rmse;
}

TEST_F(StencilGPUTest, CvStencil_NonAlignedGrid) {
  // Non-aligned to tile (32x128) but still meets TMA requirements (ldm % 16 == 0, ldm >= 128)
  constexpr size_t H = 68, W = 144;
  static_assert((H * W) % 4 == 0);
  auto input = generate_random(H * W);

  ozi::Stencil<Real, 7, 7> box;
  for (auto &c : box.coeffs)
    c = 1.0 / 49.0;
  auto setup = compute_ozaki_setup(box);

  auto result =
      run_full_cvstencil<1>(box, input, {H, W}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);
  auto gold = test_utils::compute_gold_standard(box, input, std::array<size_t, 2>{H, W});

  double rmse = test_utils::interior_relative_rmse<Real, 2>(result, gold, {H, W}, box.offsets);
  EXPECT_LT(rmse, 1e-12) << "CvStencil non-aligned RMSE: " << rmse;
}

TEST_F(StencilGPUTest, CvStencil_MinimalGrid) {
  // Minimal grid that meets TMA ldm >= 128 requirement.
  // Single output tile (32x128), exercises boundary handling on all edges.
  constexpr size_t H = 32, W = 128;
  static_assert((H * W) % 4 == 0);
  auto input = generate_random(H * W);

  ozi::Stencil<Real, 7, 7> box;
  for (auto &c : box.coeffs)
    c = 1.0 / 49.0;
  auto setup = compute_ozaki_setup(box);

  auto result =
      run_full_cvstencil<1>(box, input, {H, W}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);
  auto gold = test_utils::compute_gold_standard(box, input, std::array<size_t, 2>{H, W});

  // Only 4 interior cells
  double rmse = test_utils::interior_relative_rmse<Real, 2>(result, gold, {H, W}, box.offsets);
  EXPECT_LT(rmse, 1e-12) << "CvStencil minimal grid RMSE: " << rmse;
}

// ============================================================
// Test Suite 4: BoundaryTests
// ============================================================

TEST_F(StencilGPUTest, Boundary_StencilV1_1D) {
  constexpr size_t N = 32;
  auto input = generate_random(N);

  ozi::Stencil<Real, 3> d2x({1.0, -2.0, 1.0});
  auto setup = compute_ozaki_setup(d2x);

  auto result =
      run_full_ozaki_v1<1>(d2x, input, {N}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);

  // GPU boundary path preserves boundary cells ~= original input via emulated_gain
  double rmse = test_utils::boundary_relative_rmse<Real, 1>(result, input, {N}, d2x.offsets);
  EXPECT_LT(rmse, 1e-10) << "1D boundary RMSE: " << rmse;
}

TEST_F(StencilGPUTest, Boundary_StencilV1_2D) {
  constexpr size_t H = 16, W = 16;
  auto input = generate_random(H * W);

  ozi::Stencil<Real, 3, 3> heat({0, 3.0 / 16, 0, 3.0 / 16, 4.0 / 16, 3.0 / 16, 0, 3.0 / 16, 0});
  auto setup = compute_ozaki_setup(heat);

  auto result =
      run_full_ozaki_v1<1>(heat, input, {H, W}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);

  // GPU boundary path preserves boundary cells ~= original input via emulated_gain
  double rmse = test_utils::boundary_relative_rmse<Real, 2>(result, input, {H, W}, heat.offsets);
  EXPECT_LT(rmse, 1e-10) << "2D boundary RMSE: " << rmse;
}

TEST_F(StencilGPUTest, Boundary_StencilV1_3D) {
  constexpr size_t D = 12, H = 12, W = 12;
  static_assert((D * H * W) % 4 == 0);
  auto input = generate_random(D * H * W);

  ozi::Stencil<Real, 3, 3, 3> box;
  for (auto &c : box.coeffs)
    c = 1.0 / 27.0;
  auto setup = compute_ozaki_setup(box);

  auto result = run_full_ozaki_v1<1>(box, input, {D, H, W}, 1, setup.a_coeff, setup.scaling_factor,
                                     setup.extra_msb_shift);

  // GPU boundary path preserves boundary cells ~= original input via emulated_gain
  double rmse = test_utils::boundary_relative_rmse<Real, 3>(result, input, {D, H, W}, box.offsets);
  EXPECT_LT(rmse, 1e-10) << "3D boundary RMSE: " << rmse;
}

TEST_F(StencilGPUTest, Boundary_CvStencil_2D) {
  constexpr size_t H = 32, W = 128;
  static_assert((H * W) % 4 == 0);
  auto input = generate_random(H * W);

  ozi::Stencil<Real, 7, 7> box;
  for (auto &c : box.coeffs)
    c = 1.0 / 49.0;
  auto setup = compute_ozaki_setup(box);

  auto result =
      run_full_cvstencil<1>(box, input, {H, W}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);

  // GPU boundary path preserves boundary cells ~= original input via emulated_gain
  // For 7x7 on 16x16, most cells are boundary (interior is only 4x4)
  double rmse = test_utils::boundary_relative_rmse<Real, 2>(result, input, {H, W}, box.offsets);
  EXPECT_LT(rmse, 1e-10) << "CvStencil 2D boundary RMSE: " << rmse;
}

TEST_F(StencilGPUTest, Boundary_StencilV1_AllBoundary) {
  // Small grid with many boundary cells relative to interior
  constexpr size_t H = 4, W = 4;
  static_assert((H * W) % 4 == 0);
  auto input = generate_random(H * W);

  ozi::Stencil<Real, 3, 3> heat({0, 3.0 / 16, 0, 3.0 / 16, 4.0 / 16, 3.0 / 16, 0, 3.0 / 16, 0});
  auto setup = compute_ozaki_setup(heat);

  auto result =
      run_full_ozaki_v1<1>(heat, input, {H, W}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);

  // GPU boundary path preserves boundary cells ~= original input via emulated_gain
  // Offsets are {1,1}, so interior is [1,3) x [1,3) = 4 cells. Boundary is remaining 12 cells.
  double rmse = test_utils::boundary_relative_rmse<Real, 2>(result, input, {H, W}, heat.offsets);
  EXPECT_LT(rmse, 1e-10) << "All-boundary RMSE: " << rmse;
}

// ============================================================
// Test Suite 6: M > 1 Stencil Tests (Gap 1)
// ============================================================

TEST_F(StencilGPUTest, OzakiV1_M2_SingleStep_1D) {
  constexpr size_t N = 1024;
  auto input = generate_random(N);

  // {100, -511, 100}: a=1, max_int=511 (10 bits), M=2
  ozi::Stencil<Real, 3> s({100.0, -511.0, 100.0});
  auto setup = compute_ozaki_setup(s);
  ASSERT_EQ(setup.m, 2) << "Expected M=2 for {100,-511,100}";

  auto result =
      run_full_ozaki_v1<2>(s, input, {N}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);
  auto gold = test_utils::compute_gold_standard(s, input, std::array<size_t, 1>{N});

  double rmse = test_utils::interior_relative_rmse<Real, 1>(result, gold, {N}, s.offsets);
  EXPECT_LT(rmse, 1e-10) << "M=2 1D single-step RMSE: " << rmse;
}

TEST_F(StencilGPUTest, OzakiV1_M2_SingleStep_2D) {
  constexpr size_t H = 64, W = 64;
  auto input = generate_random(H * W);

  // 3x3 M=2 stencil: same integer structure as 1D {100,-511,100}
  ozi::Stencil<Real, 3, 3> s({0, 100, 0, 100, -511, 100, 0, 100, 0});
  auto setup = compute_ozaki_setup(s);
  ASSERT_EQ(setup.m, 2) << "Expected M=2 for 2D {100,-511,100} cross";

  auto result =
      run_full_ozaki_v1<2>(s, input, {H, W}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);
  auto gold = test_utils::compute_gold_standard(s, input, std::array<size_t, 2>{H, W});

  double rmse = test_utils::interior_relative_rmse<Real, 2>(result, gold, {H, W}, s.offsets);
  EXPECT_LT(rmse, 1e-10) << "M=2 2D single-step RMSE: " << rmse;
}

TEST_F(StencilGPUTest, OzakiV1_M3_SingleStep_1D) {
  constexpr size_t N = 1024;
  auto input = generate_random(N);

  // Coprime coefficients with max=32769 (16 bits), a=1 (abs_min=1), p_max=15, M=3
  ozi::Stencil<Real, 5> s({-1.0, 16385.0, -32769.0, 16385.0, -1.0});
  auto setup = compute_ozaki_setup(s);
  ASSERT_EQ(setup.m, 3) << "Expected M=3 for large coprime 5-point stencil";

  auto result =
      run_full_ozaki_v1<3>(s, input, {N}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);
  auto gold = test_utils::compute_gold_standard(s, input, std::array<size_t, 1>{N});

  double rmse = test_utils::interior_relative_rmse<Real, 1>(result, gold, {N}, s.offsets);
  EXPECT_LT(rmse, 1e-8) << "M=3 1D single-step RMSE: " << rmse;
}

TEST_F(StencilGPUTest, OzakiV1_M2_MultiStep5_1D) {
  constexpr size_t N = 512;
  constexpr int32_t T = 5;
  auto input = generate_random(N);

  ozi::Stencil<Real, 3> s({100.0, -511.0, 100.0});
  auto setup = compute_ozaki_setup(s);
  ASSERT_EQ(setup.m, 2);

  auto result =
      run_full_ozaki_v1<2>(s, input, {N}, T, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);
  auto gold = test_utils::compute_gold_standard_multistep(s, input, std::array<size_t, 1>{N}, T);

  double rmse = test_utils::interior_relative_rmse<Real, 1>(result, gold, {N}, s.offsets);
  EXPECT_LT(rmse, T * 1e-10) << "M=2 1D multi-step (T=5) RMSE: " << rmse;
}

TEST_F(StencilGPUTest, CvStencil_M2_SingleStep_7x7) {
  constexpr size_t H = 128, W = 128;
  auto input = generate_random(H * W);

  // 7x7 M=2 stencil: center=-511, 4 cardinal neighbors=200, rest=0
  // a=1, max_int=511 (10 bits), M=2
  ozi::Stencil<Real, 7, 7> s;
  for (auto &c : s.coeffs)
    c = 0.0;
  s.coeffs[3 * 7 + 3] = -511.0; // center
  s.coeffs[2 * 7 + 3] = 200.0;  // up
  s.coeffs[4 * 7 + 3] = 200.0;  // down
  s.coeffs[3 * 7 + 2] = 200.0;  // left
  s.coeffs[3 * 7 + 4] = 200.0;  // right
  auto setup = compute_ozaki_setup(s);
  ASSERT_EQ(setup.m, 2) << "Expected M=2 for 7x7 cardinal stencil";

  auto result =
      run_full_cvstencil<2>(s, input, {H, W}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);
  auto gold = test_utils::compute_gold_standard(s, input, std::array<size_t, 2>{H, W});

  double rmse = test_utils::interior_relative_rmse<Real, 2>(result, gold, {H, W}, s.offsets);
  EXPECT_LT(rmse, 1e-10) << "CvStencil M=2 7x7 single-step RMSE: " << rmse;
}

TEST_F(StencilGPUTest, CvStencil_M2_MatchesV1_7x7) {
  constexpr size_t H = 128, W = 128;
  auto input = generate_random(H * W);

  // Same 7x7 M=2 stencil as above
  ozi::Stencil<Real, 7, 7> s;
  for (auto &c : s.coeffs)
    c = 0.0;
  s.coeffs[3 * 7 + 3] = -511.0;
  s.coeffs[2 * 7 + 3] = 200.0;
  s.coeffs[4 * 7 + 3] = 200.0;
  s.coeffs[3 * 7 + 2] = 200.0;
  s.coeffs[3 * 7 + 4] = 200.0;
  auto setup = compute_ozaki_setup(s);
  ASSERT_EQ(setup.m, 2);

  auto result_cv =
      run_full_cvstencil<2>(s, input, {H, W}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);

  TearDown();
  SetUp();

  auto result_v1 =
      run_full_ozaki_v1<2>(s, input, {H, W}, 1, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);

  double rmse = test_utils::interior_relative_rmse<Real, 2>(result_cv, result_v1, {H, W}, s.offsets);
  EXPECT_LT(rmse, 1e-14) << "CvStencil vs StencilV1 M=2 cross-kernel RMSE: " << rmse;
}

// ============================================================
// Test Suite 7: Overflow/Shift Mechanism Tests (Gap 2)
// ============================================================

TEST_F(StencilGPUTest, OzakiV1_ShiftCount_SingleShift_Heat2D) {
  constexpr size_t H = 64, W = 64;
  constexpr int32_t T = 2;
  auto input = generate_random(H * W);

  ozi::Stencil<Real, 3, 3> heat({0, 3.0 / 16, 0, 3.0 / 16, 4.0 / 16, 3.0 / 16, 0, 3.0 / 16, 0});
  auto setup = compute_ozaki_setup(heat);

  // Heat: sf=16, log2_sf=4. T=2 gives 1 update_ozaki_desc call, which forces >= 1 shift.
  auto result =
      run_full_ozaki_v1<1>(heat, input, {H, W}, T, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);

  EXPECT_GE(h_desc.n_shifts, 1) << "Expected at least 1 shift for Heat2D T=2 (sf=16)";

  auto gold = test_utils::compute_gold_standard_multistep(heat, input, std::array<size_t, 2>{H, W}, T);
  double rmse = test_utils::interior_relative_rmse<Real, 2>(result, gold, {H, W}, heat.offsets);
  EXPECT_LT(rmse, T * 5e-12) << "Post-shift accuracy degraded: " << rmse;
}

TEST_F(StencilGPUTest, OzakiV1_ShiftCount_MultiShift_Heat2D) {
  constexpr size_t H = 64, W = 64;
  constexpr int32_t T = 10;
  auto input = generate_random(H * W);

  ozi::Stencil<Real, 3, 3> heat({0, 3.0 / 16, 0, 3.0 / 16, 4.0 / 16, 3.0 / 16, 0, 3.0 / 16, 0});
  auto setup = compute_ozaki_setup(heat);

  // Heat: sf=16, log2_sf=4. T=10: ceil(10*4/7) = ceil(5.71) = 6 proactive shifts.
  auto result =
      run_full_ozaki_v1<1>(heat, input, {H, W}, T, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);

  EXPECT_GE(h_desc.n_shifts, 6) << "Expected at least 6 shifts for Heat2D T=10 (sf=16)";

  auto gold = test_utils::compute_gold_standard_multistep(heat, input, std::array<size_t, 2>{H, W}, T);
  double rmse = test_utils::interior_relative_rmse<Real, 2>(result, gold, {H, W}, heat.offsets);
  EXPECT_LT(rmse, T * 5e-12) << "Post-multi-shift accuracy: " << rmse;
}

TEST_F(StencilGPUTest, OzakiV1_NoShift_D2x) {
  constexpr size_t N = 512;
  constexpr int32_t T = 3;
  auto input = generate_random(N);

  // {1,-2,1}: M=1, sf=1 (log2_sf=0), no proactive shifts.
  // Slice L1=4, threshold=floor(127/4)-1=30. N(0,1) data stays bounded -> no reactive shifts.
  ozi::Stencil<Real, 3> s({1.0, -2.0, 1.0});
  auto setup = compute_ozaki_setup(s);
  ASSERT_EQ(setup.m, 1);

  auto result =
      run_full_ozaki_v1<1>(s, input, {N}, T, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);

  EXPECT_EQ(h_desc.n_shifts, 0) << "Expected 0 shifts for d2x with sf=1 and low L1";

  auto gold = test_utils::compute_gold_standard_multistep(s, input, std::array<size_t, 1>{N}, T);
  double rmse = test_utils::interior_relative_rmse<Real, 1>(result, gold, {N}, s.offsets);
  EXPECT_LT(rmse, T * 5e-12) << "No-shift accuracy: " << rmse;
}

TEST_F(StencilGPUTest, OzakiV1_CarrySliceIndex_Rotates) {
  constexpr size_t H = 64, W = 64;
  constexpr int32_t T = 3;
  auto input = generate_random(H * W);

  ozi::Stencil<Real, 3, 3> heat({0, 3.0 / 16, 0, 3.0 / 16, 4.0 / 16, 3.0 / 16, 0, 3.0 / 16, 0});
  auto setup = compute_ozaki_setup(heat);

  // Heat T=3: ceil(3*4/7)=2 proactive shifts -> carry_slice_index should rotate
  auto result =
      run_full_ozaki_v1<1>(heat, input, {H, W}, T, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);

  EXPECT_GE(h_desc.n_shifts, 2) << "Expected at least 2 shifts for carry rotation test";
  EXPECT_NE(h_desc.carry_slice_index, 0) << "carry_slice_index should have rotated";

  auto gold = test_utils::compute_gold_standard_multistep(heat, input, std::array<size_t, 2>{H, W}, T);
  double rmse = test_utils::interior_relative_rmse<Real, 2>(result, gold, {H, W}, heat.offsets);
  EXPECT_LT(rmse, T * 5e-12) << "Carry rotation accuracy: " << rmse;
}

TEST_F(StencilGPUTest, OzakiV1_ReactiveOverflow_1D) {
  constexpr size_t N = 512;
  constexpr int32_t T = 5;
  auto input = generate_random(N);

  // {3, -2, 3}: a=1, sf=1 (log2_sf=0), no proactive shifts.
  // L1=8, threshold=floor(127/8)-1=14. Stencil amplifies (L1>1),
  // so reactive overflow should be triggered after a few timesteps.
  ozi::Stencil<Real, 3> s({3.0, -2.0, 3.0});
  auto setup = compute_ozaki_setup(s);

  auto result =
      run_full_ozaki_v1<1>(s, input, {N}, T, setup.a_coeff, setup.scaling_factor, setup.extra_msb_shift);

  EXPECT_GT(h_desc.n_shifts, 0) << "Expected reactive overflow shifts for amplifying stencil";

  auto gold = test_utils::compute_gold_standard_multistep(s, input, std::array<size_t, 1>{N}, T);
  double rmse = test_utils::interior_relative_rmse<Real, 1>(result, gold, {N}, s.offsets);
  EXPECT_LT(rmse, T * 1e-10) << "Reactive overflow accuracy: " << rmse;
}

TEST_F(StencilGPUTest, CvStencil_ShiftCount_Heat7x7) {
  constexpr size_t H = 128, W = 128;
  constexpr int32_t T = 5;
  auto input = generate_random(H * W);

  // Heat 7x7: 0.0234375 = 3/128 for 12 cardinal neighbors, center = 0.71875 = 92/128
  // a=1/128, int coeffs max=92 (7 bits), L1_int=12*3+92=128.
  // After sf>=128 fix: extra_msb_shift>=1, M=2, sf=64.
  // log2_sf=6. T=5: total_shifts=ceil(30/7)=5. With T-1=4 update calls, expect >= 4 shifts.
  ozi::Stencil<Real, 7, 7> heat7x7;
  for (auto &c : heat7x7.coeffs)
    c = 0.0;
  // Center
  heat7x7.coeffs[3 * 7 + 3] = 0.71875;
  // 12 cardinal neighbors (cross shape extending 3 cells each direction)
  for (int i = 0; i < 7; ++i) {
    if (i != 3) {
      heat7x7.coeffs[i * 7 + 3] = 0.0234375; // column 3
      heat7x7.coeffs[3 * 7 + i] = 0.0234375; // row 3
    }
  }
  auto setup = compute_ozaki_setup(heat7x7);
  ASSERT_EQ(setup.m, 2) << "Expected M=2 for Heat7x7";

  auto result = run_full_cvstencil<2>(heat7x7, input, {H, W}, T, setup.a_coeff, setup.scaling_factor,
                                       setup.extra_msb_shift);

  EXPECT_GE(h_desc.n_shifts, 4) << "Expected at least 4 shifts for Heat7x7 T=5 (sf=64)";

  auto gold = test_utils::compute_gold_standard_multistep(heat7x7, input, std::array<size_t, 2>{H, W}, T);
  double rmse = test_utils::interior_relative_rmse<Real, 2>(result, gold, {H, W}, heat7x7.offsets);
  EXPECT_LT(rmse, T * 1e-10) << "CvStencil heat7x7 post-shift accuracy: " << rmse;
}
