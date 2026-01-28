cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
project(ozaki-stencil VERSION 0.0.0 LANGUAGES CXX CUDA)

# settings
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# options
set(CUDA_COMPUTE_CAPABILITY "86" CACHE STRING "The CUDA Compute capability to target.")
option(BUILD_TESTING "Build the project's tests" ON)
option(OZAKI_NVTX_EVENTS "Whether to record NVTX events" ON)

# dependencies
include(cmake/Dependencies.cmake)
find_package(CUDAToolkit REQUIRED)
find_package(CUDNN REQUIRED)
# remove a needless warning
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenMP)

# common config for all targets (GPU and CPU)
add_library(ozaki_common INTERFACE)
target_include_directories(ozaki_common INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(ozaki_common INTERFACE cxx_std_20)
if(OpenMP_CXX_FOUND)
  message(STATUS "Found OpenMP. Building parallel OzakiStencil.")
  target_link_libraries(ozaki_common INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(ozaki_common INTERFACE OZAKI_USE_OPENMP)
  target_compile_options(ozaki_common INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
  )
else()
  message(STATUS "OpenMP not found. Building without OpenMP support.")
endif()

# common config for GPU targets only
add_library(ozaki_cuda_common INTERFACE)
target_link_libraries(ozaki_cuda_common INTERFACE ozaki_common ${CUDAToolkit_NVTX_LIBRARY} CUDA::cuda_driver CUDA::cudart)
set_target_properties(ozaki_cuda_common PROPERTIES
  CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY}
  CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(ozaki_cuda_common INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info --expt-relaxed-constexpr --extended-lambda>
)

if(OZAKI_NVTX_EVENTS)
  target_compile_definitions(ozaki_cuda_common INTERFACE OZAKI_NVTX_EVENTS)
endif()

# helpers for defining targets with shared boilerplate
function(add_cpu_lib name)
  add_library(${name} ${ARGN})
  target_link_libraries(${name} PUBLIC ozaki_common)
endfunction()

function(add_cpu_app name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE ozaki_common Boost::program_options)
endfunction()

function(add_gpu_lib name)
  add_library(${name} ${ARGN})
  target_link_libraries(${name} PUBLIC ozaki_cuda_common)
  set_target_properties(${name} PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY}
    CUDA_SEPARABLE_COMPILATION ON
  )
endfunction()

function(add_gpu_app name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE ozaki_cuda_common Boost::program_options)
  set_target_properties(${name} PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY}
    CUDA_SEPARABLE_COMPILATION ON
  )
endfunction()

# subfolders
add_subdirectory(lib)
add_subdirectory(benchmark)

# testing
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

