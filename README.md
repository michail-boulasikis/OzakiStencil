# Stencil Ozaki

Code for implementing iterative stencil loops using the Ozaki scheme.
High precision (FP64, FP128 and so on) is emulated using INT8 on both CPU and GPU for improved performance and energy efficiency.

## Requirements

A C++17 capable compiler, a CUDA Toolkit, CMake.

## Build, Test & Run

Run `bash run.sh`. This (a) configures the repository, (b) builds the repository and (c) runs the test using `ctest`. For a list of options you can supply to the script run `bash run.sh -h`.
