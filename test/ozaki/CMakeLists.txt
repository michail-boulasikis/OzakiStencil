include(GoogleTest)

add_cpu_app(test-ozaki-cpu test_ozaki_cpu.cpp)
add_gpu_app(test-ozaki-gpu test_ozaki_gpu.cu)
target_link_libraries(test-ozaki-cpu PRIVATE Eigen3::Eigen GTest::gtest_main)
target_link_libraries(test-ozaki-gpu PRIVATE ozigpu CUDA::cublas CUDA::cudart GTest::gtest_main)

gtest_discover_tests(test-ozaki-cpu)
gtest_discover_tests(test-ozaki-gpu)
