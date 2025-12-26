include(GoogleTest)

add_gpu_app(test-gpu test_gpu.cu)
target_link_libraries(test-gpu PRIVATE CUDA::cudart CUDA::cuda_driver GTest::gtest_main gpuozi)

add_cpu_app(test-stencil test_stencil.cpp)
target_link_libraries(test-stencil PRIVATE GTest::gtest_main)

 gtest_discover_tests(test-gpu)
gtest_discover_tests(test-stencil)
