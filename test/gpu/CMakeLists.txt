include(GoogleTest)

add_gpu_app(test-ozaki-gpu test_ozaki_gpu.cu)
target_link_libraries(test-ozaki-gpu PRIVATE ozigpu CUDA::cudart GTest::gtest_main quadmath)
target_compile_options(test-ozaki-gpu PRIVATE -fext-numeric-literals)

add_gpu_app(test-stencil-gpu test_stencil_gpu.cu)
target_link_libraries(test-stencil-gpu PRIVATE ozigpu CUDA::cudart GTest::gtest_main quadmath stencil-registry)

gtest_discover_tests(test-ozaki-gpu)
gtest_discover_tests(test-stencil-gpu)
