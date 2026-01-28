import argparse
import cuda.bindings.nvrtc as nvrtc
import cuda.bindings.driver as driver
import numpy as np

from stencil import (
    Stencil,
    StencilKernelGenerator,
    ptx_kernel_from_stencil,
    run_benchmark,
    DTYPE_MAP,
)

if __name__ == "__main__":
    stencils = {
        "box2d1r": Stencil(np.random.rand(3, 3), np.ones(2)),
        "box2d2r": Stencil(np.random.rand(5, 5), 2 * np.ones(2)),
        "box2d3r": Stencil(np.random.rand(7, 7), 3 * np.ones(2)),
        "heat2d1r": Stencil(
            np.array([[0.0, 0.2, 0.0], [0.2, 0.4, 0.2], [0.0, 0.2, 0.0]]), np.ones(2)
        ),
        "heat2d2r": Stencil(
            np.array(
                [
                    [0.0, 0.0, 0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.1, 0.0, 0.0],
                    [0.1, 0.1, 0.4, 0.1, 0.1],
                    [0.0, 0.0, 0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.1, 0.0, 0.0],
                ]
            ),
            2 * np.ones(2),
        ),
        "heat2d3r": Stencil(
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                    [0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1],
                    [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                ]
            ),
            3 * np.ones(2),
        ),
    }

    # Argparse to select specific stencil
    parser = argparse.ArgumentParser(
        description="Compile and Benchmark Stencil Kernels"
    )
    parser.add_argument(
        "--stencil",
        type=str,
        required=True,
        choices=stencils.keys(),
        help="Name of the stencil to run",
    )
    parser.add_argument(
        "--arch",
        type=int,
        required=True,
        choices=[70, 80, 86, 90, 100, 120],
        help="Compute capability of targeted GPU",
    )
    parser.add_argument(
        "--input-size", type=int, required=True, help="Input dimension size"
    )
    args = parser.parse_args()

    # Select the requested stencil
    sname = args.stencil
    s = stencils[sname]
    input_shape = [args.input_size, args.input_size]

    # 1. Generate Code
    print(f"Generating PTX for {sname}...")
    code = ptx_kernel_from_stencil(s, input_shape, kernel_name=sname)

    # 2. Compile PTX
    err, prog = nvrtc.nvrtcCreateProgram(
        code.encode("utf-8"), f"{sname}.cu".encode(), 0, [], []
    )
    # Add optimization flags if desired, e.g., ["-O3"]
    (err,) = nvrtc.nvrtcCompileProgram(
        prog, 1, [f"--gpu-architecture=sm_{args.arch}".encode()]
    )
    # Check for compile errors
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        err, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * log_size
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"Compilation failed: {log.decode()}")

    err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
    ptx = b" " * ptx_size
    nvrtc.nvrtcGetPTX(prog, ptx)

    # 3. Run Benchmark
    # We infer dtype from the generator logic (float64 default in numpy if not specified)
    gen = StencilKernelGenerator(s, input_shape, kernel_name=sname)
    dtype = DTYPE_MAP[gen.c_type]  # Map back from C-string to numpy type

    run_benchmark(ptx, sname, input_shape, dtype)
