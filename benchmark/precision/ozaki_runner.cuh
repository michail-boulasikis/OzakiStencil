#ifndef HEADER_OZAKI_RUNNER_CUH
#define HEADER_OZAKI_RUNNER_CUH

#include <ozaki/cpu/ozi.hpp>
#include <ozaki/gpu/benchmark_utils.cuh>
#include <ozaki/gpu/gpu_checks.cuh>
#include <ozaki/gpu/ozi.cuh>
#include <ozaki/gpu/ozi_kernels.cuh>
#include <ozaki/gpu/ozi_stencil_plain.cuh>
#include <ozaki/utils/tensor.hpp>

#include <algorithm>
#include <climits>
#include <cuda_runtime_api.h>
#include <format>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifdef __GNUC__
#include <quadmath.h>
#endif

using Real = double;
using AccumulatorType = __float128;

// Non-templated base class for Ozaki stencil runners.
// Holds all CUDA resources with RAII cleanup.
class OzakiRunnerBase {
protected:
  static constexpr int RDC_THREADS = 512;
  static constexpr size_t STENCIL_BLOCK_DIM_X = 32;
  static constexpr size_t STENCIL_BLOCK_DIM_Y = 16;

  int8_t *d_slices_1 = nullptr, *d_slices_2 = nullptr;
  ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>> *d_desc = nullptr;
  ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>> h_desc;
  Real *d_data = nullptr;
  AccumulatorType *d_res = nullptr;

  Tensor<Real> host_input;
  int32_t num_slices_total = 0;
  size_t n_elems = 0;
  int32_t n_iterations;
  std::string file_prefix;
  HighPrecType scaling_factor;

  ozigpu::L2CacheInvalidator cache_invalidator;

public:
  OzakiRunnerBase(const Tensor<Real> &input, int32_t iterations, int32_t extra_msb, int32_t extra_lsb,
                  const std::string &prefix, HighPrecType a)
      : host_input(input), n_elems(input.size()), n_iterations(iterations), file_prefix(prefix), scaling_factor(a) {
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

  OzakiRunnerBase(const OzakiRunnerBase &) = delete;
  OzakiRunnerBase &operator=(const OzakiRunnerBase &) = delete;

  void allocate_resources() {
    NVTX_PUSH("CUDA Alloc & Upload");
    CUDA_CHECK(cudaMalloc(&d_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>)));
    CUDA_CHECK(cudaMalloc(&d_data, n_elems * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_res, n_elems * sizeof(AccumulatorType)));
    CUDA_CHECK(cudaMemcpy(d_data, host_input.data.data(), n_elems * sizeof(Real), cudaMemcpyHostToDevice));
    NVTX_POP();
  }

  void compute_dynamic_range(int32_t extra_msb = 0, int32_t extra_lsb = 0) {
    NVTX_PUSH("Compute Dyn Range");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    h_desc = ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>();
    CUDA_CHECK(cudaMemcpy(d_desc, &h_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>),
                          cudaMemcpyHostToDevice));

    const int32_t nThreads = RDC_THREADS;
    const int32_t nBlocks = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / RDC_THREADS;

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
    const int32_t nThreads = std::min(static_cast<size_t>(RDC_THREADS), n_elems / 4);
    const int32_t nBlocks = ((n_elems / 4) + nThreads - 1) / nThreads;

    ozigpu::slice<ozigpu::OzakiFP_Int8<Real>><<<nBlocks, nThreads>>>(d_slices_1, d_data, d_desc, n_elems);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(
        cudaMemcpy(d_slices_2, d_slices_1, num_slices_total * n_elems * sizeof(int8_t), cudaMemcpyDeviceToDevice));
    NVTX_POP();
  }

  virtual std::vector<AccumulatorType> run_experiment(int32_t active_slices, const std::string &output_path,
                                                      bool save_tensors) = 0;

  int get_max_slices() const { return num_slices_total; }
};

// Concrete runner parameterized on stencil slices (M) and stencil dimensions (Dims...).
template <int32_t M, size_t... Dims> class OzakiRunnerDirect : public OzakiRunnerBase {
  using IntStencilType = ozi::Stencil<int8_t, Dims...>;
  ozi::StencilSlices<IntStencilType, M> stencil_slices_;

public:
  template <typename CoeffT>
  OzakiRunnerDirect(const ozi::Stencil<CoeffT, Dims...> &stencil, const Tensor<Real> &input, int32_t iterations,
                    int32_t extra_msb, int32_t extra_lsb, const std::string &prefix, HighPrecType a,
                    HighPrecType a_coeff, int32_t extra_msb_shift = 0)
      : OzakiRunnerBase(input, iterations, extra_msb, extra_lsb, prefix, a) {
    // Compute int8_t stencil slices: divide coefficients by a_coeff to get integer stencil.
    stencil_slices_ =
        ozicpu::slice_stencil<ozicpu::OzakiFP_Int8<Real>, ozi::Stencil<CoeffT, Dims...>, M>(stencil, a_coeff,
                                                                                             extra_msb_shift);
  }

  std::vector<AccumulatorType> run_experiment(int32_t active_slices, const std::string &output_path,
                                              bool save_tensors) override {
    NVTX_PUSH(std::format("Run {} Slices (Direct)", active_slices).c_str());

    // Compute overflow threshold from the most significant stencil slice
    double slice0_l1 = stencil_slices_.slices[0].template l1_norm<double>();
    int32_t l1_int = static_cast<int32_t>(slice0_l1);
    if (l1_int == 0)
      l1_int = 1;

    h_desc.n_slices_input = active_slices;
    h_desc.shift_required = 0;
    h_desc.carry_slice_index = 0;
    h_desc.overflow_danger_threshold = (std::numeric_limits<int8_t>::max() / l1_int) - 1;
    h_desc.just_shifted = 1;
    h_desc.n_shifts = 0;

    CUDA_CHECK(cudaMemcpy(d_desc, &h_desc, sizeof(ozigpu::OzakiDescriptor<ozigpu::OzakiFP_Int8<Real>>),
                          cudaMemcpyHostToDevice));

    prepare_slices();
    cache_invalidator.invalidate();

    // Configure block and grid dimensions based on stencil dimensionality
    constexpr size_t ndims = sizeof...(Dims);
    dim3 block, grid;

    if constexpr (ndims == 1) {
      block = dim3(256);
      grid = dim3((host_input.shape[0] + 255) / 256);
    } else if constexpr (ndims == 2) {
      block = dim3(STENCIL_BLOCK_DIM_X, STENCIL_BLOCK_DIM_Y);
      grid = dim3((host_input.shape[0] + block.x - 1) / block.x, (host_input.shape[1] + block.y - 1) / block.y);
    } else {
      block = dim3(8, 8, 4);
      grid = dim3((host_input.shape[0] + 7) / 8, (host_input.shape[1] + 7) / 8, (host_input.shape[2] + 3) / 4);
    }

    // Construct input shape array for the kernel
    typename IntStencilType::shape_type input_shape{};
    for (size_t d = 0; d < ndims && d < host_input.shape.size(); ++d)
      input_shape[d] = host_input.shape[d];

    NVTX_PUSH("Integration Loop");
    for (int32_t t = 0; t < n_iterations; ++t) {
      ozigpu::stencil_v1<ozigpu::OzakiFP_Int8<Real>, M, Dims...>
          <<<grid, block>>>(d_slices_2, d_slices_1, stencil_slices_, input_shape, d_desc);

      if (t < n_iterations - 1) {
        ozigpu::update_ozaki_desc<<<1, 32>>>(d_desc);
      }

      std::swap(d_slices_1, d_slices_2);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    NVTX_POP();

    // Mend slices back to floating point and apply scale factor
    const int32_t nThreadsMend = std::min(static_cast<size_t>(RDC_THREADS), n_elems / 4);
    const int32_t nBlocksMend = ((n_elems / 4) + nThreadsMend - 1) / nThreadsMend;

    const int32_t nThreadsScale = std::min(static_cast<size_t>(RDC_THREADS), n_elems);
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
    long double factor = ozigpu::total_integration_scale_factor_quad<ozigpu::OzakiFP_Int8<Real>>(
        this->scaling_factor, n_iterations, h_desc.n_shifts);
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

#endif // HEADER_OZAKI_RUNNER_CUH
