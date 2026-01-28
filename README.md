# OzakiStencil

## Project Description

This is an exploration of what happens when you apply the Ozaki scheme to stencil computations. The Ozaki scheme breaks down high-precision numbers (FP64, FP128) into multiple INT8 slices, essentially reconstructing precision through decomposition and careful accumulation. The idea here is to see if this approach can work for iterative stencil operations, using INT8 arithmetic on tensor cores while still maintaining the numerical properties needed for things like PDE solvers.

The implementation includes both CPU and GPU variants (using CUDA), so you can experiment with how the approach behaves across different hardware.

## Requirements & Build

### Requirements

**Essential:**
- C++20 capable compiler (GCC 10+, Clang 12+, or MSVC 19.30+)
- CMake 3.23+
- CUDA Toolkit 11.0+ (for GPU support)
- Boost (program options library for command-line utilities)

**Optional (but recommended):**
- OpenMP (for CPU parallelization - auto-detected)
- Google Test (gtest) for running unit tests
- Eigen for linear algebra operations in tests/benchmarks (used as a reference)

### Build & Test

Run `bash run.sh` to:
1. Configure the repository with CMake
2. Build the library and tests
3. Run tests with `ctest`

For build options, run `bash run.sh -h`.

## How OzakiStencil Works

### The Basic Idea

The Ozaki scheme breaks a high-precision number into multiple INT8 "slices":
```
value = Σ(slice_i × 2^exponent_i)  for i = 0 to N
```

Each slice is an INT8 value (-127 to 127), and together they capture the full precision. For stencil computations, the approach is:

1. **Decompose** the input data into INT8 slices
2. **Apply the stencil** to each slice independently (using INT8 arithmetic)
3. **Accumulate** the partial results with appropriate scaling
4. **Reconstruct** the final answer by combining all slices back together

So far I have proven that you can get FP64 or even higher precision using this scheme. But it remains to be seen if performance is faster or whether more optimizations are needed.

### Implementation Details

**CPU side** (`include/ozaki/cpu/`):
- Slice decomposition using frexp and careful residual computation
- Mending (reconstruction) by scaling and summing slices
- Stencil kernels that work on the decomposed data
- Optional OpenMP parallelization for slice-by-slice processing

**GPU side** (`include/ozaki/gpu/`):
- Dynamic range analysis to figure out how many slices are actually needed
- CUDA kernels for decomposition and reconstruction
- Stencil kernels that apply operations to INT8 data
- Hysteresis and carry management to track numerical state across iterations

The GPU implementation is more experimental: it tracks slice shifts and overflow information as iterations progress, trying to maintain precision without exploding the number of slices needed.
It also manages to avoid slicing and mending at every time step by carefully computing the slices by shifting and accumulating.
