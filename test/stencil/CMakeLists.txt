include(GoogleTest)

add_cpu_app(test-stencil test_stencil.cpp)
target_link_libraries(test-stencil PRIVATE GTest::gtest_main)

gtest_discover_tests(test-stencil)
