add_gpu_app(slice-precision main.cu)
target_compile_options(slice-precision PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fext-numeric-literals>)
target_link_libraries(slice-precision PRIVATE CUDA::cudart CUDA::cuda_driver ozigpu quadmath)
