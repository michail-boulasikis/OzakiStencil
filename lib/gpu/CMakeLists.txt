add_gpu_lib(ozaki-gpu
  sliced_tensor.cu
  packed_stencil.cu
  stencil_integrator.cu
  stencil_2d.cu
  stencil_3d.cu
)
# The M=9 5x5x5 kernel keeps seven diagonals in registers; two CTAs per SM
# need at most 120 registers per thread.
set_source_files_properties(stencil_3d.cu PROPERTIES
  COMPILE_OPTIONS "-Xptxas=-maxrregcount=120,-override-directive-values"
)
target_include_directories(ozaki-gpu SYSTEM PRIVATE ${CUTLASS_INCLUDE_DIR})
target_link_libraries(ozaki-gpu PUBLIC stencil-registry PRIVATE CUDA::cudart CUDA::cuda_driver)
