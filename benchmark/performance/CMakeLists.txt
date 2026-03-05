add_gpu_app(bench-stencil bench_stencil.cu)
target_compile_options(bench-stencil PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fext-numeric-literals>)
target_link_libraries(bench-stencil PRIVATE CUDA::cudart CUDA::cuda_driver ozigpu quadmath)
