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
    # Dtype mapping from string to numpy dtype
    dtype_map = {
        "float32": np.float32,
        "float64": np.float64,
        "float128": np.longdouble,
    }

    # Argparse to select specific stencil
    parser = argparse.ArgumentParser(
        description="Compile and Benchmark Stencil Kernels"
    )
    parser.add_argument(
        "--stencil",
        type=str,
        required=True,
        choices=["box2d1r", "box2d2r", "box2d3r", "heat2d1r", "heat2d2r", "heat2d3r"],
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
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=["float32", "float64", "float128"],
        help="Data type for stencil coefficients",
    )
    args = parser.parse_args()

    # Select the data type and create stencils
    dtype = dtype_map[args.dtype]

    stencils = {
        "box2d1r": Stencil(np.random.rand(3, 3).astype(dtype), np.ones(2, dtype=dtype)),
        "box2d2r": Stencil(
            np.random.rand(5, 5).astype(dtype), (2 * np.ones(2)).astype(dtype)
        ),
        "box2d3r": Stencil(
            np.random.rand(7, 7).astype(dtype), (3 * np.ones(2)).astype(dtype)
        ),
        "heat2d1r": Stencil(
            np.array([[0.0, 0.2, 0.0], [0.2, 0.4, 0.2], [0.0, 0.2, 0.0]], dtype=dtype),
            np.ones(2, dtype=dtype),
        ),
        "heat2d2r": Stencil(
            np.array(
                [
                    [0.0, 0.0, 0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.1, 0.0, 0.0],
                    [0.1, 0.1, 0.4, 0.1, 0.1],
                    [0.0, 0.0, 0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.1, 0.0, 0.0],
                ],
                dtype=dtype,
            ),
            (2 * np.ones(2)).astype(dtype),
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
                ],
                dtype=dtype,
            ),
            (3 * np.ones(2)).astype(dtype),
        ),
    }

    # Select the requested stencil
    sname = args.stencil
    s = stencils[sname]
    input_shape = [args.input_size, args.input_size]
    print(f"Using dtype: {args.dtype}")

    # 1. Generate Code
    print(f"Generating PTX for {sname}...")
    code = ptx_kernel_from_stencil(s, input_shape, kernel_name=sname)

    # 2. Compile PTX
    err, prog = nvrtc.nvrtcCreateProgram(
        code.encode("utf-8"), f"{sname}.cu".encode(), 0, [], []
    )
    # Add optimization flags if desired, e.g., ["-O3"]
    (err,) = nvrtc.nvrtcCompileProgram(
        prog,
        2,
        [
            f"--gpu-architecture=sm_{args.arch}".encode(),
            "--device-float128".encode(),
        ],
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

    run_benchmark(ptx, sname, input_shape, dtype, profiled_steps=1000)
