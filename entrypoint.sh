#!/bin/bash
set -e 

echo "========================================"
echo " Preparing Artifact Environment"
echo "========================================"

# Check if the evaluator passed the GPU architecture variable
if [ -z "$CUDA_COMPUTE_CAPABILITY" ]; then
    echo "Warning: CUDA_COMPUTE_CAPABILITY environment variable not set."
    echo "Defaulting to '120' (sm120). If this is wrong,"
    echo "run the container with: -e CUDA_COMPUTE_CAPABILITY=\"your_arch\""
    CUDA_COMPUTE_CAPABILITY="120"
fi

export CUDA_COMPUTE_CAPABILITY="${CUDA_COMPUTE_CAPABILITY}"

./build.sh -c -t -- -DCUDA_COMPUTE_CAPABILITY="${CUDA_COMPUTE_CAPABILITY}"

echo ""
echo "========================================"
echo "OzakiStencil has been built and environment has been set up."
echo "You are now inside the OzakiStencil artifact container. You can run the following experiments:"
echo ""
echo "1. Correctness Tests:"
echo " To run the random stencil suite:"
echo "    python scripts/precision_tester.py"
echo " To generate correctness plots:"
echo "   ./correctness.sh"
echo "   (Runs the default 15-stencil suite; pass stencil files to override it.)"
echo ""
echo "2. Performance Benchmarks:"
echo "   ./performance.sh"
echo ""
echo "Outputs and plots will be saved to the /artifact/data/results directory."
echo "========================================"

exec /bin/bash
