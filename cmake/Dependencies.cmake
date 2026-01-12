include(FetchContent)

if(BUILD_TESTING)
  message(STATUS "Fetching Test Dependencies...")
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
  )
  FetchContent_MakeAvailable(googletest)

  set(EIGEN_BUILD_TESTING OFF)
  set(EIGEN_BUILD_PKGCONFIG OFF)
  set(EIGEN_BUILD_DOC OFF)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen
    GIT_TAG 5.0.1
  )
  FetchContent_MakeAvailable(eigen)
endif()
