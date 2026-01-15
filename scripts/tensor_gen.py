import numpy as np
import struct
import argparse
import sys


def get_dtype_map():
    dtypes = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
    }
    if hasattr(np, "float128"):
        dtypes["float128"] = np.float128
    elif hasattr(np, "float96"):
        dtypes["float128"] = np.float96
    else:
        dtypes["float128"] = np.longdouble
    return dtypes


def generate_data(n, n_peaks, dtype_name, halo_avg, output_path):
    dtype_map = get_dtype_map()
    if dtype_name not in dtype_map:
        print(f"Error: {dtype_name} not supported.")
        sys.exit(1)

    target_dtype = dtype_map[dtype_name]
    print(f"Generating data strictly in: {target_dtype}")

    dim = np.linspace(0, 1, n).astype(target_dtype)
    X, Y = np.meshgrid(dim, dim)

    peak_means_x = np.random.uniform(0.3, 0.7, n_peaks).astype(target_dtype)
    peak_means_y = np.random.uniform(0.3, 0.7, n_peaks).astype(target_dtype)
    peak_std = np.random.uniform(0.01, 0.001, n_peaks).astype(target_dtype)

    output = np.zeros_like(X, dtype=target_dtype)

    for px, py, std in zip(peak_means_x, peak_means_y, peak_std):
        term = np.exp(-((X - px) ** 2 + (Y - py) ** 2) / std)
        output += term
    if halo_avg:
        avg = np.mean(output)
        output[0, :] = avg
        output[-1, :] = avg
        output[:, 0] = avg
        output[:, -1] = avg

    header = struct.pack(f"<{'Q' * (1 + len(output.shape))}", 2, n, n)
    with open(output_path, "wb") as outf:
        outf.write(header)
        output.tofile(outf)

    print(f"\tTensor |min| = {np.min(np.abs(output))}")
    print(f"\tTensor |max| = {np.max(np.abs(output))}")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", type=str)
    parser.add_argument(
        "--type",
        type=str,
        default="float64",
        choices=["float16", "float32", "float64", "float128"],
    )
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--halo-avg", action="store_true")
    args = parser.parse_args()

    generate_data(args.N, 10, args.type, args.halo_avg, args.output_file)
