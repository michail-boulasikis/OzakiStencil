add_gpu_lib(ozigpu ozi.cu)
target_link_libraries(ozigpu PRIVATE CUDA::cudart CUDA::cuda_driver)
