add_gpu_app(bench-cudnn main.cu)
target_link_libraries(bench-cudnn PRIVATE CUDNN::CUDNN)
