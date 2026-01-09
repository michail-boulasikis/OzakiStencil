#include <array>
#include <fstream>
#include <iostream>
#include <ozaki/core/stencil.hpp>
#include <ozaki/cpu/ozi.hpp>
#include <ozaki/gpu/ozi.cuh>
#include <vector>

#ifndef RDC_THREADS_PER_BLOCK
#define RDC_THREADS_PER_BLOCK 512
#endif

using real = double;

std::pair<std::vector<size_t>, std::vector<real>> read_tensor_from_file(std::ifstream &f) {
  size_t n_dims;
  f.read(reinterpret_cast<char *>(&n_dims), sizeof(size_t));
  std::vector<size_t> shape(n_dims);
  f.read(reinterpret_cast<char *>(shape.data()), n_dims * sizeof(size_t));
  size_t n_elems = 1;
  for (const auto &dim : shape)
    n_elems *= dim;
  std::vector<real> data(n_elems);
  f.read(reinterpret_cast<char *>(data.data()), n_elems * sizeof(real));
  return {shape, data};
}

int main(int argc, char *argv[]) {
  NVTX_PUSH("I/O");
  std::ifstream input_data_file{argv[1], std::ios::in | std::ios::binary};
  const auto &[shape, data] = read_tensor_from_file(input_data_file);
  NVTX_POP();

  NVTX_PUSH("Stencil Slicing");
  const real r = 3.0 / 16.0;
  const std::array<real, 9> coeffs{0.0, r, 0.0, r, 1.0 - 4 * r, r, 0.0, r, 0.0};
  std::span<const real> coeffs_span{coeffs.begin(), coeffs.end()};
  const auto stencil_slices = ozicpu::slice<ozicpu::OzakiFP_Int8<real>>(coeffs_span, 1);

  const ozi::Stencil<real, 3, 3> heat{coeffs, {1, 1}};
  const ozi::Stencil<int8_t, 3, 3> heat_ozaki{std::span{stencil_slices.values.data(), coeffs.size()}};
  NVTX_POP();

  std::cout << "Binary digits of `long double` on this CPU: " << std::numeric_limits<long double>::digits << "\n";
  NVTX_PUSH("Gold Standard");
  // double buffering
  std::vector<long double> u_gold_1(data.size());
  std::vector<long double> u_gold_2(data.size());
  std::transform(data.cbegin(), data.cend(), u_gold_1.begin(), [](real x) { return static_cast<long double>(x); });

  ozi::Stencil<long double, 3, 3> heat_high;
  heat_high.offsets = heat.offsets;
  for (size_t i = 0; i < heat.coeffs.size(); ++i) {
    heat_high.coeffs[i] = static_cast<long double>(heat.coeffs[i]);
  }

  for (size_t i = 0; i < 128; ++i) {
    heat_high.template apply_to<long double>(u_gold_2, u_gold_1, {3, 3});
    std::swap(u_gold_1, u_gold_2);
  } // reference solution in u_gold_2
  NVTX_POP();

  NVTX_PUSH("CUDA Memory Setup");
  int8_t *d_slices_1, *d_slices_2;
  int32_t *d_dyn_range, h_dyn_range[64];
  real *d_data, *d_res;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  CUDA_CHECK(cudaMalloc(&d_dyn_range, 64 * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_data, data.size() * sizeof(real)));
  CUDA_CHECK(cudaMalloc(&d_res, data.size() * sizeof(real)));
  CUDA_CHECK(cudaMemcpy(d_data, data.data(), data.size() * sizeof(real), cudaMemcpyHostToDevice));
  const int32_t nThreadsReduce = RDC_THREADS_PER_BLOCK;
  const int32_t nBlocksReduce = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount / RDC_THREADS_PER_BLOCK;
  ozigpu::dyn_range<<<nBlocksReduce, nThreadsReduce>>>(d_dyn_range, d_data, data.size());
  CUDA_CHECK(cudaMemcpy(h_dyn_range, d_dyn_range, 64 * sizeof(int32_t), cudaMemcpyDeviceToHost));
  const int32_t num_slices = 1 + (h_dyn_range[0] - h_dyn_range[1]) / (CHAR_BIT * sizeof(int8_t) - 1);
  CUDA_CHECK(cudaMalloc(&d_slices_1, num_slices * data.size() * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&d_slices_2, num_slices * data.size() * sizeof(int8_t)));
  NVTX_POP();

  NVTX_PUSH("Data Slicing");
  const int32_t nThreadsMap = std::min(static_cast<size_t>(RDC_THREADS_PER_BLOCK), data.size() / 4);
  const int32_t nBlocksMap = ((data.size() / 4) + nThreadsMap - 1) / nThreadsMap;
  ozigpu::slice<ozigpu::OzakiFP_Int8<real>>
      <<<nBlocksMap, nThreadsMap>>>(d_slices_1, d_data, d_dyn_range, num_slices, data.size());
  CUDA_CHECK(cudaDeviceSynchronize());
  NVTX_POP();

  // 5. for 1 to N slices compute solution using ozaki (time using NVTX)
  NVTX_PUSH("Ozaki Stencil");
  const int32_t nThreadsStencil = 512;
  const int32_t nBlocksStencil = (data.size() + nThreadsStencil - 1) / nThreadsStencil;
  for (size_t s = 1; s < num_slices; ++s) {
    NVTX_PUSH(std::format("{} Slices", s));
    for (size_t t = 0; t < 128; ++t) {
      ozigpu::stencil_v1<ozigpu::OzakiFP_Int8<real>, 3, 3>
          <<<nBlocksStencil, nThreadsStencil>>>(d_slices_2, d_slices_1, heat_ozaki, {shape[0], shape[1]}, s);
      std::swap(d_slices_1, d_slices_2);
    }
    NVTX_POP();
    ozigpu::mend<ozigpu::OzakiFP_Int8<real>>
        <<<nBlocksMap, nThreadsMap>>>(d_res, d_slices_2, d_dyn_range, s, data.size());
    // TODO: ERROR CHECKING
    ozigpu::slice<ozigpu::OzakiFP_Int8<real>>
        <<<nBlocksMap, nThreadsMap>>>(d_slices_1, d_data, d_dyn_range, num_slices, data.size());
  }
  NVTX_POP();

  cudaFree(d_slices_2);
  cudaFree(d_slices_1);
  cudaFree(d_dyn_range);
  cudaFree(d_data);
  cudaFree(d_res);

  return 0;
}
