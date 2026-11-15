// Host entry points of the stencil kernels (one step: preamble + compute).
#ifndef OZAKI_GPU_STENCIL_KERNELS_CUH
#define OZAKI_GPU_STENCIL_KERNELS_CUH

#include <ozaki/gpu/sliced_tensor.cuh>
#include <ozaki/gpu/packed_stencil.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace ozaki::gpu {

// Computes output = stencil(input) on the interior; the halo of output is
// never written.
cudaError_t launch_stencil_2d(PackedStencil &stencil, SlicedTensor &output, SlicedTensor &input, cudaStream_t stream);
cudaError_t launch_stencil_3d(PackedStencil &stencil, SlicedTensor &output, SlicedTensor &input, cudaStream_t stream);

// TMA descriptors.
CUresult encode_input_tensor_maps_2d(const PackedStencil &stencil, SlicedTensor &tensor,
                                     std::vector<CUtensorMap> &maps);
CUresult encode_input_tensor_maps_3d(const PackedStencil &stencil, SlicedTensor &tensor,
                                     std::vector<CUtensorMap> &maps);

} // namespace ozaki::gpu

#endif
