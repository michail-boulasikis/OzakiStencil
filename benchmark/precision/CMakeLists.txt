# CPU-only: produce reference tensor files
add_cpu_app(compute-reference compute_reference.cpp)
target_link_libraries(compute-reference PRIVATE quadmath stencil-spec stencil-registry)

# GPU: run Ozaki stencil integration
add_gpu_app(run-ozaki run_ozaki.cu)
target_link_libraries(run-ozaki PRIVATE CUDA::cudart CUDA::cuda_driver ozaki-gpu quadmath stencil-spec stencil-registry)

# CPU-only: compute RMSE between reference and result tensors
add_cpu_app(compute-errors compute_errors.cpp)
target_link_libraries(compute-errors PRIVATE quadmath stencil-spec)

# CPU-only: generate FP128 Gaussian-peak tensor files
add_cpu_app(gen-tensor gen_tensor.cpp)
target_link_libraries(gen-tensor PRIVATE quadmath stencil-spec)
