# OzakiStencil

An implementation of the Ozaki scheme for high-precision iterative stencil computations on GPUs. OzakiStencil decomposes high-precision floating-point numbers (FP64, FP128) into multiple INT8 slices and applies stencil operations to each slice independently using fast INT8 tensor core arithmetic, then reconstructs the result with full precision through careful accumulation and carry propagation.

This approach enables FP64-accurate (and beyond) stencil computations using the INT8 throughput of modern GPU tensor cores, which is an order of magnitude higher than native FP64 throughput on consumer GPUs.

## Features

- **Multi-precision support**: FP64 (`double`) and FP128 (`__float128`) input data types
- **GPU tensor core acceleration**: INT8 MMA (matrix multiply-accumulate) via inline PTX on NVIDIA GPUs
- **Multi-dimensional stencils**: 1D, 2D, and 3D stencils with compile-time shape specialization
- **Two GPU kernel implementations**:
  - `stencil_v1`: General-purpose stencil kernel supporting arbitrary shapes (1D/2D/3D)
  - `cvstencil`: Tensor-core-optimized kernel for 2D 7x7 stencils using the ConvStencil dual-tessellation approach
- **Automatic precision management**: Dynamic range tracking, overflow detection, and slice count adjustment across iterations
- **Integer basis optimization**: When stencil coefficients lie in a discrete subgroup of the reals, the scheme exploits this structure to reduce the number of required slices
- **CPU reference implementation**: With optional OpenMP parallelization for validation

## Requirements

**Required:**
- C++20 compiler (GCC 10+)
- CMake 3.23+
- CUDA Toolkit 12.0+
- Boost (program_options)
- cuDNN

**Optional:**
- OpenMP (CPU parallelization, auto-detected)
- Google Test (unit tests, auto-fetched by CMake)
- Boost.Multiprecision (high-precision scale factors, see below)

## Build

```bash
# Basic build and test
sh run.sh -t

# With a specific GPU architecture and high-precision scale factors
sh run.sh -t -- -DCUDA_COMPUTE_CAPABILITY="80;90;100;120" -DOZAKI_USE_BOOST_MULTIPRECISION=ON
```

Supported GPU architectures: `sm_80` (Ampere), `sm_90` (Hopper), `sm_100` (Blackwell datacenter), `sm_120` (Blackwell consumer). Multiple architectures can be specified as a semicolon-separated list for fat binary builds.

Run `sh run.sh -h` for all build options.

### High-Precision Scale Factors

At the end of integration, the result is scaled by `1/a^timesteps` where `a` is the stencil's L1 norm. When `a` is not a power of 2, computing this scale factor with `__float128` loses precision and becomes the dominant source of error.

Enable Boost.Multiprecision to compute scale factors with 500-bit precision:

```bash
cmake -B build -DOZAKI_USE_BOOST_MULTIPRECISION=ON
```

This only affects a single CPU computation and has negligible performance impact.

## Usage

### Running the Precision Benchmark

Three binaries form a sequential pipeline for measuring numerical accuracy:

```bash
# 1. Generate a high-precision reference solution
./build/bin/compute-reference -i data/2D_128x128.dat \
    --stencil-file data/stencils/3x3Lm2_0.stc -T 10

# 2. Run the Ozaki stencil integration
./build/bin/run-ozaki -i data/2D_128x128.dat \
    --stencil-file data/stencils/3x3Lm2_0.stc -T 10 --save-tensors --slice-limit 18

# 3. Compute RMSE between reference and Ozaki results
./build/bin/compute-errors -e data/tensors/ref_Heat_fp128_10.dat \
    -a data/tensors/oz_Heat_10_18.dat
```

### Running the Performance Benchmark

```bash
./build/bin/bench-stencil -S 4096x4096 -s 9 \
    --stencil-file data/stencils/7x7Lm2_0.stc -W 2 -T 40 -R 5
```

### Running Tests

```bash
# All tests
ctest --test-dir build

# Specific test suites
./build/bin/test-ozaki-cpu --gtest_filter=TestSuiteName.TestName
./build/bin/test-ozaki-gpu --gtest_filter=TestSuiteName.TestName
./build/bin/test-slice-stencil --gtest_filter=TestSuiteName.TestName
```

## How It Works

### The Ozaki Scheme for Stencils

The Ozaki scheme represents a high-precision value as a sum of INT8 slices:

```
value = sum(slice_i * 2^exponent_i)  for i = 0..N
```

Each slice is an INT8 value (-127 to 127). The number of slices N is determined dynamically from the data's exponent range. For stencil computations:

1. **Decompose** input data into INT8 slices based on dynamic range analysis
2. **Apply the stencil** to each data slice independently using INT8 arithmetic
3. **Propagate carries** between slice positions using a sliding window accumulator
4. **Reconstruct** the result by scaling and summing all output slices

The key insight is that stencil application on INT8 slices can leverage tensor core MMA instructions, which offer significantly higher throughput than scalar FP64 operations on modern GPUs.

### Avoiding Per-Step Decomposition

Rather than decomposing and reconstructing at every timestep, the GPU implementation maintains the INT8 slice representation across iterations. An `OzakiDescriptor` tracks per-integration state (exponent range, slice count, overflow flags) and triggers re-decomposition only when numerical overflow is detected, amortizing the decomposition cost.

## Project Structure

```
include/ozaki/
  core/           # Type traits, stencil representation, concepts
  cpu/            # CPU implementation (slice, mend, stencil)
  gpu/            # GPU kernels and runtime state management
  utils/          # Stencil registry, spec parsing, tensor I/O
benchmark/
  performance/    # GPU kernel timing benchmarks
  precision/      # Numerical accuracy measurement pipeline
  cudnn/          # cuDNN convolution comparison benchmark
test/
  core/           # Stencil and tensor I/O tests
  cpu/            # CPU Ozaki correctness tests
  gpu/            # GPU kernel correctness tests
data/
  stencils/       # Stencil specification files (.stc format)
scripts/          # Plotting and analysis utilities
```

### Stencil Specification Format

Stencils are defined in `.stc` text files with the following format:

```
name Heat
shape 3 3
coefficients 0 1 0 1 -4 1 0 1 0
```

Inline specifications are also supported: `"3x3:0,1,0,1,-4,1,0,1,0"`.

### Binary Tensor Format

Input/output data uses a binary tensor format: `[ndim, type, dim0, ..., dimN, values...]`

- `ndim`: number of dimensions (uint64)
- `type`: 0=float, 1=double, 2=long double, 3=__float128 (uint64)
- `dimX`: size of each dimension (uint64)
- `values`: tensor data in row-major order

## License

This project is research software. See LICENSE for details.
