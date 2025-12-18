#!/bin/bash
# Exit immediately on error, fail on unset variables, and fail pipelines
set -euo pipefail

# --- Configuration (Defaults) ---
BUILD_DIR="build"
BUILD_TYPE="Release"
CLEAN_BUILD=0 # 0 = no (incremental), 1 = yes (full clean)
RUN_TESTS=0 # 0 = no, 1 = yes
# Array to store extra arguments for CMake
CMAKE_ARGS=()

# --- Help Message ---
print_usage() {
  echo "Usage: $(basename "$0") [options] -- [cmake_args]"
  echo "Options:"
  echo "  -c, --clean    Perform a clean build, removing the build directory first."
  echo "  -g, --debug    Set build type to Debug (default: Release)."
  echo "  -t, --test     Run tests using ctest."
  echo "  -h, --help     Show this help message."
  echo "All arguments after '--' (or any unrecognized arguments) are passed directly to CMake."
  echo "Example: $(basename "$0") --clean -- -G Ninja -DMY_OPTION=ON"
}

# --- Argument Parsing ---
# Loop until all positional parameters are shifted
while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--clean)
      CLEAN_BUILD=1
      shift
      ;;
    -t|--test)
      RUN_TESTS=1
      shift
      ;;
    -g|--debug)
      BUILD_TYPE="Debug"
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    --) # End of options
      shift # Remove the '--'
      CMAKE_ARGS+=("$@") # Add all remaining args to CMAKE_ARGS
      break # Stop parsing
      ;;
    *) # Unknown option
      CMAKE_ARGS+=("$1") # Assume it's a CMake argument
      shift 
      ;;
  esac
done

# Get number of cores for parallel build. Fallback to 1.
JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
echo "Using $JOBS parallel jobs."

# --- 1. Clean (if requested) ---
if [ "$CLEAN_BUILD" -eq 1 ] && [ -d "$BUILD_DIR" ]; then
  echo "--- 🧹 Cleaning build directory ($BUILD_DIR) ---"
  rm -rf "$BUILD_DIR"
fi

# --- 2. Configure ---
CMAKE_STATUS=0
echo "--- ⚙️  Configuring (CMake) [Type: $BUILD_TYPE] ---"
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" "${CMAKE_ARGS[@]}"
CMAKE_STATUS=$?
if [ "$CMAKE_STATUS" -ne 0 ]; then
  echo "--- ❌ Configuration incomplete, errors occurred (Exit code: $CMAKE_STATUS) ---"
  exit "$CMAKE_STATUS"
fi

# --- 3. Build ---
CMAKE_STATUS=0
echo "--- 🏗️  Building (Parallel: $JOBS jobs) ---"
cmake --build "$BUILD_DIR" --parallel "$JOBS"
CMAKE_STATUS=$?
if [ "$CMAKE_STATUS" -ne 0 ]; then
  echo "--- ❌ Configuration complete, but build FAILED (Exit code: $CMAKE_STATUS) ---"
  exit "$CMAKE_STATUS"
fi

# --- 4. Test (if requested) ---
if [ "$RUN_TESTS" -eq 1 ]; then
  echo "--- 🧪 Testing (CTest) ---"

  # Initialize test status to 0 (success)
  CTEST_STATUS=0
  # Run ctest. If it fails (non-zero exit), capture the status code in CTEST_STATUS.
  # The '|| CTEST_STATUS=$?' trick prevents 'set -e' from exiting the script here.
  (cd "$BUILD_DIR" && ctest --output-on-failure) || CTEST_STATUS=$?

# --- 5. Final Status ---
# Check the test status and print the final message
  if [ "$CTEST_STATUS" -ne 0 ]; then
    echo "--- ❌ Build complete, but Tests FAILED (Exit code: $CTEST_STATUS) ---"
    # Exit with the original failure code so CI/CD pipelines fail correctly
    exit "$CTEST_STATUS"
  else
    echo "--- ✅ Build and Test Complete ---"
    # Exit with 0 for success
    exit 0
  fi
fi

echo "--- ✅ Build Complete ---"
exit 0
