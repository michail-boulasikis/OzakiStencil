cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
project(ozaki-stencil VERSION 0.0.0 LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CUDA_COMPUTE_CAPABILITY "86" CACHE STRING "The CUDA Compute capability to target.")
option(BUILD_TESTING "Build the project's tests" ON)
option(OZAKI_NVTX_EVENTS "Whether to record NVTX events for use when profiling with Nsight Systems" ON)

# Dependencies
find_package(CUDAToolkit REQUIRED)
find_package(CUDNN REQUIRED)

# Helpers for defining targets with shared boilerplate
function(add_cpu_lib name)
  add_library(${name} ${ARGN})
  target_include_directories(${name} PUBLIC ${PROJECT_SOURCE_DIR}/include)
  target_compile_features(${name} PRIVATE cxx_std_20)
  set_target_properties(${name} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/lib")
endfunction()

function(add_cpu_app name)
  add_executable(${name} ${ARGN})  
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_features(${name} PRIVATE cxx_std_20)
  set_target_properties(${name} PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endfunction()

function(add_gpu_lib name)
  add_library(${name} ${ARGN})
  target_include_directories(${name} PUBLIC ${PROJECT_SOURCE_DIR}/include)
  target_compile_features(${name} PRIVATE cxx_std_20)
  target_compile_definitions(${name} PRIVATE OZAKI_NVTX_EVENTS)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --generate-line-info --expt-relaxed-constexpr --extended-lambda>)
  target_link_libraries(${name} PRIVATE ${CUDAToolkit_NVTX_LIBRARY})
  set_target_properties(${name} PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY}
    CUDA_SEPARABLE_COMPILATION ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/lib")
endfunction()

function(add_gpu_app name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name} PUBLIC ${PROJECT_SOURCE_DIR}/include)
  target_compile_definitions(${name} PRIVATE OZAKI_NVTX_EVENTS)
  target_compile_features(${name} PRIVATE cxx_std_20)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --generate-line-info --expt-relaxed-constexpr --extended-lambda>)
  target_link_libraries(${name} PRIVATE ${CUDAToolkit_NVTX_LIBRARY})
  set_target_properties(${name} PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY}
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endfunction()

# Subfolders
add_subdirectory(lib)
add_subdirectory(benchmark)

# Testing
if(BUILD_TESTING)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
  )
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen
    GIT_TAG 5.0.1
  )
  FetchContent_MakeAvailable(googletest)
  set(EIGEN_BUILD_TESTING OFF)
  set(EIGEN_BUILD_PKGCONFIG OFF)
  set(EIGEN_BUILD_DOC OFF)
  FetchContent_MakeAvailable(Eigen)
  add_subdirectory(test)
endif()

