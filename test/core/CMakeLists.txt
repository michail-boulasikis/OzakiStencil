include(GoogleTest)

add_cpu_app(test-stencil test_stencil.cpp)
target_link_libraries(test-stencil PRIVATE GTest::gtest_main)

add_cpu_app(test-stencil-registry test_stencil_registry.cpp)
target_link_libraries(test-stencil-registry PRIVATE GTest::gtest_main quadmath)

add_cpu_app(test-stencil-spec test_stencil_spec.cpp)
target_link_libraries(test-stencil-spec PRIVATE GTest::gtest_main quadmath stencil-spec)

add_cpu_app(test-tensor-io test_tensor_io.cpp)
target_link_libraries(test-tensor-io PRIVATE GTest::gtest_main quadmath)

gtest_discover_tests(test-stencil)
gtest_discover_tests(test-stencil-registry)
gtest_discover_tests(test-stencil-spec)
gtest_discover_tests(test-tensor-io)
