include(GoogleTest)

add_cpu_app(test-ozaki-cpu test_ozaki_cpu.cpp)
add_cpu_app(test-slice-stencil test_slice_stencil.cpp)

target_link_libraries(test-ozaki-cpu PRIVATE GTest::gtest_main quadmath)
target_link_libraries(test-slice-stencil PRIVATE GTest::gtest_main quadmath)

gtest_discover_tests(test-ozaki-cpu)
gtest_discover_tests(test-slice-stencil)
