#!/usr/bin/env python3
"""
Run the full precision benchmark pipeline for a set of stencils.

For each stencil x timestep combination:
  1. compute-reference  -> FP128 reference tensor
  2. run-ozaki          -> Ozaki tensors for s=2..18
  3. compute-errors     -> RMSE between reference and each Ozaki result

Usage:
    python scripts/run_precision_sweep.py data/stencils/7x7Lm1_0.stc data/stencils/7x7Nm1_0.stc
    python scripts/run_precision_sweep.py data/stencils/7x7*.stc -o results.csv
    python scripts/run_precision_sweep.py data/stencils/7x7Lm1_0.stc -i myinput.dat
"""

import argparse
import csv
import glob
import os
import struct
import subprocess
import sys

import numpy as np

BUILD_BIN = "./build/bin"
TENSOR_DIR = "./data/tensors"

DEFAULT_TIMESTEPS = [1, 10, 100, 1000]
DEFAULT_SLICE_MIN = 2
DEFAULT_SLICE_MAX = 18

DEFAULT_INPUT_SHAPES = {
    1: [256],
    2: [128, 128],
    3: [32, 32, 32],
}


def parse_stencil_file(path):
    """Extract name and dimensionality from a .stc file."""
    name = None
    shape = None
    with open(path) as f:
        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "name":
                name = tokens[1]
            elif tokens[0] == "shape":
                shape = [int(x) for x in tokens[1:]]
    if name is None:
        raise ValueError(f"No 'name' field in {path}")
    if shape is None:
        raise ValueError(f"No 'shape' field in {path}")
    return name, len(shape)


def write_tensor_file(path, data):
    """Write a tensor in the project's binary format (type_id=1 for double)."""
    data = np.asarray(data, dtype=np.float64)
    shape = data.shape
    ndim = len(shape)
    header = struct.pack(f"<{'Q' * (2 + ndim)}", ndim, 1, *shape)
    with open(path, "wb") as f:
        f.write(header)
        data.tofile(f)


def generate_input(ndim, seed=42):
    """Generate a random input tensor for the given dimensionality."""
    rng = np.random.default_rng(seed)
    shape = DEFAULT_INPUT_SHAPES[ndim]
    data = rng.standard_normal(shape)
    path = os.path.join(TENSOR_DIR, f"sweep_input_{ndim}d.dat")
    write_tensor_file(path, data)
    return path


def run_cmd(args, label=""):
    """Run a subprocess, print failures to stderr. Returns (ok, stdout, stderr)."""
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        last = (
            result.stderr.strip().splitlines()[-1]
            if result.stderr.strip()
            else "(no output)"
        )
        print(f"  [FAIL] {label}: {last}", file=sys.stderr)
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()


def cleanup_tensors(stencil_name, T, tensor_dir, slice_min, slice_max):
    """Remove intermediate tensor files for a given stencil/T combination."""
    ref = os.path.join(tensor_dir, f"ref_{stencil_name}_fp128_{T}.dat")
    if os.path.exists(ref):
        os.remove(ref)
    for s in range(slice_min, slice_max + 1):
        oz = os.path.join(tensor_dir, f"oz_{stencil_name}_{T}_{s}.dat")
        if os.path.exists(oz):
            os.remove(oz)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stencils", nargs="+", help="Stencil .stc file paths")
    parser.add_argument(
        "-i", "--input", help="Input tensor .dat file (auto-generated if omitted)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="precision_sweep.csv",
        help="Output CSV (default: precision_sweep.csv)",
    )
    parser.add_argument(
        "-T",
        "--timesteps",
        nargs="+",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"Timesteps (default: {DEFAULT_TIMESTEPS})",
    )
    parser.add_argument(
        "--slice-min",
        type=int,
        default=DEFAULT_SLICE_MIN,
        help=f"Min data slices (default: {DEFAULT_SLICE_MIN})",
    )
    parser.add_argument(
        "--slice-max",
        type=int,
        default=DEFAULT_SLICE_MAX,
        help=f"Max data slices (default: {DEFAULT_SLICE_MAX})",
    )
    parser.add_argument("--scaled", action="store_true", help="Use scaled RMSE")
    parser.add_argument(
        "-A",
        "--algorithm",
        default="direct",
        choices=["direct", "convstencil"],
        help="Algorithm (default: direct)",
    )
    parser.add_argument(
        "--input-shape",
        help="Input shape override, e.g. '128,128' or '32,32,32'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for input generation (default: 42)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove intermediate tensor files after each stencil/T combo",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append results to an existing CSV instead of overwriting",
    )
    args = parser.parse_args()

    os.makedirs(TENSOR_DIR, exist_ok=True)

    # Parse all stencil files to validate and get dimensionalities
    stencil_info = []
    for stc_path in args.stencils:
        try:
            name, ndim = parse_stencil_file(stc_path)
            stencil_info.append((stc_path, name, ndim))
        except (ValueError, FileNotFoundError) as e:
            print(f"Skipping {stc_path}: {e}", file=sys.stderr)

    if not stencil_info:
        print("No valid stencil files provided.", file=sys.stderr)
        sys.exit(1)

    # Prepare input tensors (one per dimensionality)
    input_by_ndim = {}
    if args.input:
        for _, _, ndim in stencil_info:
            input_by_ndim[ndim] = args.input
    else:
        for _, _, ndim in stencil_info:
            if ndim in input_by_ndim:
                continue
            if args.input_shape:
                shape = [int(x) for x in args.input_shape.split(",")]
                if len(shape) != ndim:
                    print(
                        f"Warning: --input-shape has {len(shape)} dims, "
                        f"stencil needs {ndim}D; using default",
                        file=sys.stderr,
                    )
                    input_by_ndim[ndim] = generate_input(ndim, seed=args.seed)
                else:
                    rng = np.random.default_rng(args.seed)
                    data = rng.standard_normal(shape)
                    tag = "x".join(str(s) for s in shape)
                    path = os.path.join(TENSOR_DIR, f"sweep_input_{tag}.dat")
                    write_tensor_file(path, data)
                    input_by_ndim[ndim] = path
            else:
                input_by_ndim[ndim] = generate_input(ndim, seed=args.seed)
            print(f"Input tensor ({ndim}D): {input_by_ndim[ndim]}")

    total = len(stencil_info) * len(args.timesteps)
    done = 0

    if args.append:
        file_exists = os.path.exists(args.output) and os.path.getsize(args.output) > 0
        csvfile = open(args.output, "a", newline="")
    else:
        file_exists = False
        csvfile = open(args.output, "w", newline="")

    with csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["stencil", "T", "slices", "rmse"])

        for stc_path, stencil_name, ndim in stencil_info:
            input_file = input_by_ndim.get(ndim)
            if input_file is None:
                print(
                    f"Skipping {stencil_name}: no input tensor for {ndim}D",
                    file=sys.stderr,
                )
                continue

            for T in args.timesteps:
                done += 1
                print(f"[{done}/{total}] {stencil_name}  T={T}")

                # Step 1: compute-reference (FP128)
                ok, _, _ = run_cmd(
                    [
                        f"{BUILD_BIN}/compute-reference",
                        "-i",
                        input_file,
                        "--stencil-file",
                        stc_path,
                        "-T",
                        str(T),
                        "--dir",
                        TENSOR_DIR + "/",
                        "--prefix",
                        "ref",
                    ],
                    f"compute-reference {stencil_name} T={T}",
                )
                if not ok:
                    writer.writerow([stencil_name, T, "", "FAIL_REF"])
                    csvfile.flush()
                    continue

                # Step 2: run-ozaki (all slices at once)
                ok, _, _ = run_cmd(
                    [
                        f"{BUILD_BIN}/run-ozaki",
                        "-i",
                        input_file,
                        "--stencil-file",
                        stc_path,
                        "-T",
                        str(T),
                        "--slice-limit",
                        str(args.slice_max),
                        "--save-tensors",
                        "--dir",
                        TENSOR_DIR + "/",
                        "--prefix",
                        "oz",
                        "--algorithm",
                        args.algorithm,
                    ],
                    f"run-ozaki {stencil_name} T={T}",
                )
                if not ok:
                    writer.writerow([stencil_name, T, "", "FAIL_OZ"])
                    csvfile.flush()
                    if args.clean:
                        cleanup_tensors(
                            stencil_name,
                            T,
                            TENSOR_DIR,
                            args.slice_min,
                            args.slice_max,
                        )
                    continue

                # Step 3: compute-errors for each slice count
                ref_file = os.path.join(TENSOR_DIR, f"ref_{stencil_name}_fp128_{T}.dat")
                for s in range(args.slice_min, args.slice_max + 1):
                    oz_file = os.path.join(TENSOR_DIR, f"oz_{stencil_name}_{T}_{s}.dat")
                    if not os.path.exists(oz_file):
                        writer.writerow([stencil_name, T, s, "MISSING"])
                        continue

                    err_cmd = [
                        f"{BUILD_BIN}/compute-errors",
                        "-e",
                        ref_file,
                        "-a",
                        oz_file,
                    ]
                    if args.scaled:
                        err_cmd.append("--scaled")

                    ok, stdout, _ = run_cmd(
                        err_cmd, f"compute-errors {stencil_name} T={T} s={s}"
                    )
                    if ok:
                        try:
                            rmse = float(stdout)
                            writer.writerow([stencil_name, T, s, rmse])
                        except ValueError:
                            writer.writerow([stencil_name, T, s, f"PARSE_ERR:{stdout}"])
                    else:
                        writer.writerow([stencil_name, T, s, "FAIL_ERR"])

                csvfile.flush()

                if args.clean:
                    cleanup_tensors(
                        stencil_name,
                        T,
                        TENSOR_DIR,
                        args.slice_min,
                        args.slice_max,
                    )

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
