# OzakiStencil

Ozaki scheme-like arbitrary precision stencil computations on GPUs. OzakiStencil decomposes high-precision floating-point numbers (FP64, FP128) into multiple INT8 slices and applies stencil operations to each slice independently using fast INT8 tensor core arithmetic, then reconstructs the result with full precision through careful accumulation and carry propagation.

This approach enables FP64-accurate (and beyond) stencil computations using the INT8 throughput of modern GPU tensor cores.

## Evaluation

Use the Dockerfile to evaluate the claims made in the paper.
Ensure that you have docker installed, as well as the `nvidia-container-toolkit`.
Run the following command to build the image.

```bash
docker build -t ozaki_stencil .
```
If you run into network issues try using the flag `--network=host`. If the package mirror does not work, comment out the `RUN sed` lines in the Dockerfile.

The image may then be launched as follows:

```bash
docker run -it --gpus all -e CUDA_COMPUTE_CAPABILITY="90a" -v $(pwd)/data/results:/artifact/data/results ozaki_stencil
```

Make sure to replace the `CUDA_COMPUTE_CAPABILITY` environment variable with the one that corresponds to the test environment's GPU. OzakiStencil supports `90`, `90a`, `100`, `100a` and `120`.
Note that if you want to see correctness results past `double` precision, you will need to use a GPU that supports the `__float128` type (`sm100` and above). This is a limitation of the implementation, not OzakiStencil.
The volume `-v` argument allows accessing the resulting files from outside the container. 

The container is going to run `entrypoint.sh`, which will provide a shell and additional instructions on how to run the experiments.

## License

This project is research software licensed under GPL. See LICENSE for details.
