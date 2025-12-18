# - Try to find cuDNN
# Once done, this will define:
#   CUDNN_FOUND          - True if cuDNN was found
#   CUDNN_INCLUDE_DIRS   - Path to cudnn.h
#   CUDNN_LIBRARIES      - Path to libcudnn.so or cudnn.lib
#   CUDNN_VERSION        - Parsed version from cudnn_version.h

find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS
      $ENV{CUDNN_ROOT}
      $ENV{CUDA_PATH}
      /usr/local/cuda
    PATH_SUFFIXES include)

find_library(CUDNN_LIBRARY cudnn
    HINTS
      $ENV{CUDNN_ROOT}
      $ENV{CUDA_PATH}
      /usr/local/cuda
    PATH_SUFFIXES lib lib64)

set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})

# --- Parse version from cudnn_version.h ---
set(_cudnn_version_header "${CUDNN_INCLUDE_DIR}/cudnn_version.h")
if (EXISTS "${_cudnn_version_header}")
    file(READ "${_cudnn_version_header}" _cudnn_ver_content)
    string(REGEX MATCH "#define CUDNN_MAJOR[ \t]+([0-9]+)" _major_line "${_cudnn_ver_content}")
    string(REGEX MATCH "#define CUDNN_MINOR[ \t]+([0-9]+)" _minor_line "${_cudnn_ver_content}")
    string(REGEX MATCH "#define CUDNN_PATCHLEVEL[ \t]+([0-9]+)" _patch_line "${_cudnn_ver_content}")

    if (_major_line)
        string(REGEX REPLACE ".*#define CUDNN_MAJOR[ \t]+([0-9]+).*" "\\1" CUDNN_MAJOR "${_major_line}")
        string(REGEX REPLACE ".*#define CUDNN_MINOR[ \t]+([0-9]+).*" "\\1" CUDNN_MINOR "${_minor_line}")
        string(REGEX REPLACE ".*#define CUDNN_PATCHLEVEL[ \t]+([0-9]+).*" "\\1" CUDNN_PATCHLEVEL "${_patch_line}")
        set(CUDNN_VERSION "${CUDNN_MAJOR}.${CUDNN_MINOR}.${CUDNN_PATCHLEVEL}")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN
    REQUIRED_VARS CUDNN_LIBRARY CUDNN_INCLUDE_DIR
    VERSION_VAR CUDNN_VERSION)

if (CUDNN_FOUND AND NOT TARGET CUDNN::CUDNN)
    add_library(CUDNN::CUDNN UNKNOWN IMPORTED)
    set_target_properties(CUDNN::CUDNN PROPERTIES
        IMPORTED_LOCATION "${CUDNN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}")
endif()
