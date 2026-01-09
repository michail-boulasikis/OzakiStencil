add_gpu_app(slice-precision main.cu)
target_link_libraries(slice-precision PRIVATE CUDA::cudart CUDA::cuda_driver ozigpu)
