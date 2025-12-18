add_gpu_lib(gpuozi kernels.cu)
target_link_libraries(gpuozi PRIVATE CUDA::cudart CUDA::cuda_driver)
