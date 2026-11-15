include(GoogleTest)

add_gpu_app(test-stencil-gpu test_stencil_gpu.cu)
target_link_libraries(test-stencil-gpu PRIVATE ozaki-gpu CUDA::cudart GTest::gtest_main quadmath stencil-registry)
gtest_discover_tests(test-stencil-gpu)
