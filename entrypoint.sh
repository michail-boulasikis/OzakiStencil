#!/bin/bash
set -e 

echo "========================================"
echo " Preparing Artifact Environment"
echo "========================================"

# Check if the evaluator passed the GPU architecture variable
if [ -z "$CUDA_COMPUTE_CAPABILITY" ]; then
    echo "Warning: CUDA_COMPUTE_CAPABILITY environment variable not set."
    echo "Defaulting to '90a' (Hopper). If this is wrong,"
    echo "run the container with: -e CUDA_COMPUTE_CAPABILITY=\"your_arch\""
    CUDA_COMPUTE_CAPABILITY="90a"
fi

export CUDA_COMPUTE_CAPABILITY="${CUDA_COMPUTE_CAPABILITY}"

./build.sh -c -t -- -DCUDA_COMPUTE_CAPABILITY="${CUDA_COMPUTE_CAPABILITY}" -DOZAKI_USE_BOOST_MULTIPRECISION="ON"

echo ""
echo "========================================"
echo "OzakiStencil has been built and environment has been set up."
echo "You are now inside the OzakiStencil artifact container. You can run the following experiments:"
echo ""
echo "1. Correctness Tests:"
echo "   ./correctness.sh <stencils...>"
echo "   (Example (Figure 6 in paper): ./correctness.sh data/stencils/heat.stc data/stencils/ad.stc data/stencils/7x7box.stc data/stencils/3x3x3box.stc)"
echo ""
echo "2. Performance Benchmarks:"
echo "   ./performance.sh"
echo ""
echo "Outputs and plots will be saved to the /artifact/data/results directory."
echo "========================================"

exec /bin/bash
