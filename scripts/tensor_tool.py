import numpy as np
import struct
import argparse
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns


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


def get_type_id_from_dtype(dtype):
    """Maps numpy dtype to type ID (0=float, 1=double, 2=long double, 3=__float128)."""
    dtype_map = get_dtype_map()
    if dtype == dtype_map["float32"]:
        return 0
    elif dtype == dtype_map["float64"]:
        return 1
    elif dtype == dtype_map["float128"]:
        return 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def load_tensor(filepath):
    """Reads a binary tensor file and infers type from file."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    dtype_map = get_dtype_map()
    type_id_map = {
        0: dtype_map["float32"],
        1: dtype_map["float64"],
        2: dtype_map["float128"],
        3: dtype_map["float128"],
    }

    with open(filepath, "rb") as f:
        n_dims_bytes = f.read(8)
        if not n_dims_bytes:
            return None
        n_dims = struct.unpack("Q", n_dims_bytes)[0]

        type_id_bytes = f.read(8)
        type_id = struct.unpack("Q", type_id_bytes)[0]

        if type_id not in type_id_map:
            print(f"Error: Unknown type ID {type_id} in {filepath}")
            return None

        dtype = type_id_map[type_id]

        dims_bytes = f.read(8 * n_dims)
        dims = struct.unpack("Q" * n_dims, dims_bytes)
        data_bytes = f.read()
        tensor_data = np.frombuffer(data_bytes, dtype=dtype)
        try:
            tensor = tensor_data.reshape(dims)
        except ValueError as e:
            print(
                f"Error reshaping data in {filepath}. Expected {dims}, got {tensor_data.shape}"
            )
            raise e
        print(f"Loaded {os.path.basename(filepath)}: Shape={tensor.shape}")
    return tensor


def plot_tensor_simple(tensor, title):
    """Plots a tensor using seaborn."""
    if tensor is None:
        print("No tensor to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ndim = tensor.ndim

    if ndim == 1:
        sns.lineplot(data=tensor, ax=ax)
        ax.set_title(f"{title} (1D)")
    elif ndim == 2:
        sns.heatmap(tensor, ax=ax, cmap="viridis", annot=False)
        ax.set_title(f"{title}")
    else:
        ax.text(0.5, 0.5, f"{ndim}D Tensor\nNot Supported", ha="center", va="center")
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def generate_data(n, n_peaks, dtype_name, halo_avg, output_path):
    dtype_map = get_dtype_map()
    if dtype_name not in dtype_map:
        print(f"Error: {dtype_name} not supported.")
        sys.exit(1)

    target_dtype = dtype_map[dtype_name]
    type_id = get_type_id_from_dtype(target_dtype)
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

    header = struct.pack(f"<{'Q' * (2 + len(output.shape))}", 2, type_id, n, n)
    with open(output_path, "wb") as outf:
        outf.write(header)
        output.tofile(outf)

    print(f"\tTensor |min| = {np.min(np.abs(output))}")
    print(f"\tTensor |max| = {np.max(np.abs(output))}")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument(
        "--plot", "-p", action="store_true", help="Plot mode: load and display a tensor"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="float64",
        choices=["float16", "float32", "float64", "float128"],
    )
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--halo-avg", action="store_true")
    args = parser.parse_args()

    if args.plot:
        tensor = load_tensor(args.file)
        if args.halo_avg:
            print("WARNING: Halo averaging mode ignored when in plot mode.")
        if tensor is not None:
            plot_tensor_simple(tensor, os.path.basename(args.file))
    else:
        generate_data(args.N, 10, args.type, args.halo_avg, args.file)
