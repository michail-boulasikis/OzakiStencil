add_gpu_app(bench-stencil bench_stencil.cu)
target_link_libraries(bench-stencil PRIVATE CUDA::cudart CUDA::cuda_driver ozaki-gpu quadmath stencil-spec stencil-registry)
