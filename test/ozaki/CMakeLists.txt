include(GoogleTest)

add_cpu_app(test-ozaki test_ozaki.cpp)
target_link_libraries(test-ozaki PRIVATE ozicpu Eigen3::Eigen GTest::gtest_main)

gtest_discover_tests(test-ozaki)
