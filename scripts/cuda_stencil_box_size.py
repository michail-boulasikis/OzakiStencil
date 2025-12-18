import argparse
import cuda.bindings.nvrtc as nvrtc 
import cuda.bindings.driver as driver
import numpy as np

from stencil import Stencil, StencilKernelGenerator, ptx_kernel_from_stencil, run_benchmark, check_cuda_err

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Box stencil kernel generator and profiler")
    parser.add_argument("-r", "--radius", type=int, required=True, help="The radius of the box stencil")
    parser.add_argument("-i", "--input-size", type=int, required=True, help="Size of the input array")
    parser.add_argument("--arch", type=int, required=True, choices=[70, 80, 86, 90, 100], help="Compute capability of targeted GPU")
    parser.add_argument("-w", "--warmup-runs", type=int, default=10, help="Runs to execute before actually profiling")
    parser.add_argument("-n", "--runs", type=int, default=128, help="Runs to execute when profiling")
    args = parser.parse_args()

    stencil_size = args.radius * 2 + 1
    stencil = Stencil(np.random.rand(stencil_size, stencil_size), (args.radius + 1) * np.ones(2))
    input_shape = [args.input_size, args.input_size]

    code = ptx_kernel_from_stencil(stencil, input_shape, kernel_name=f"box2d{args.radius}")
    err, prog = nvrtc.nvrtcCreateProgram(code.encode("utf-8"), f"box2d{args.radius}.cu".encode(), 0, [], [])
    err, = nvrtc.nvrtcCompileProgram(prog, 1, [f"--gpu-architecture=sm_{args.arch}".encode()])
    # Check for compile errors
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        err, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * log_size
        nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"Compilation failed: {log.decode('utf-8')}")

    err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
    ptx = b" " * ptx_size
    nvrtc.nvrtcGetPTX(prog, ptx)

    if args.warmup_runs + args.runs != 0:
        run_benchmark(ptx, f"box2d{args.radius}", input_shape, np.float64, args.warmup_runs, args.runs)


