import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, PercentFormatter
from tensor_tool import load_tensor

# Configuration
INPUT_FOLDER = "./data/"
FILE_PATTERN = "test_oz_*.dat"
REFERENCES = [
    {"name": f"{INPUT_FOLDER}test_ref_float.dat", "dtype": np.float32},
    {"name": f"{INPUT_FOLDER}test_ref_double.dat", "dtype": np.float64},
]


def plot_tensor(tensor, ax, title, log_scale=False):
    if tensor is None:
        return

    ndim = tensor.ndim

    if ndim == 1:
        if log_scale:
            sns.lineplot(data=tensor, ax=ax)
            ax.set_yscale("log")
            ax.set_title(f"{title} (1D - Log Scale)")
        else:
            sns.lineplot(data=tensor, ax=ax)
            ax.set_title(f"{title} (1D)")
    elif ndim == 2:
        if log_scale:
            masked_tensor = np.ma.masked_equal(tensor, 0)
            if masked_tensor.count() == 0:
                vmin = 1e-10
                norm = None
                print(f"Warning: {title} is all zeros.")
            else:
                vmin = masked_tensor.min()
                vmax = masked_tensor.max()
                norm = LogNorm(vmin=vmin, vmax=vmax)

            sns.heatmap(
                tensor,
                ax=ax,
                cmap="rocket",
                annot=False,
                norm=norm,
                cbar_kws={"ticks": LogLocator()},
            )
            ax.set_title(f"{title} (Log Scale)")
        else:
            sns.heatmap(tensor, ax=ax, cmap="viridis", annot=False)
            ax.set_title(f"{title}")
    else:
        ax.text(0.5, 0.5, f"{ndim}D Tensor\nNot Supported", ha="center", va="center")


def plot_error_histogram(tensor, ax, title):
    data = tensor.flatten()
    valid_data = data[np.isfinite(data) & (data > 0)]

    if len(valid_data) == 0:
        ax.text(0.5, 0.5, "Errors are 0 or NaN", ha="center", va="center")
        ax.set_title(title)
        return

    min_val = np.min(valid_data)
    max_val = np.max(valid_data)

    if min_val == max_val:
        bins = [min_val * 0.9, max_val * 1.1]
    else:
        bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)

    weights = np.ones_like(valid_data) / len(data)
    ax.hist(
        valid_data,
        bins=bins,
        weights=weights,
        color="#d62728",
        edgecolor="white",
        alpha=0.7,
    )

    ax.set_xscale("log")
    ax.set_yscale("linear")

    ax.yaxis.set_major_formatter(PercentFormatter(1))

    ax.set_xlabel("Relative Error Magnitude")
    ax.set_ylabel("% of Points")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    stats_text = (
        f"Min: {min_val:.1e}\nMed: {np.median(valid_data):.1e}\nMax: {max_val:.1e}"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )


def get_largest_index_file(folder, pattern):
    search_path = os.path.join(folder, pattern)
    candidates = glob.glob(search_path)
    if not candidates:
        return None

    regex = re.compile(r"(\d+)")
    max_idx = -1
    best_file = None

    for fname in candidates:
        match = regex.search(os.path.basename(fname))
        if match:
            curr_idx = int(match.group(1))
            if curr_idx > max_idx:
                max_idx = curr_idx
                best_file = fname
    return best_file


def main():
    target_file = get_largest_index_file(INPUT_FOLDER, FILE_PATTERN)
    if not target_file:
        print("No files found matching the pattern.")
        return

    print(f"Target: {target_file}")
    target_tensor = load_tensor(target_file, dtype=np.float32)
    if target_tensor is None:
        return

    plot_items = []

    plot_items.append(
        (f"Result: {os.path.basename(target_file)}", target_tensor, "spatial")
    )

    for ref_data in REFERENCES:
        ref_tensor = load_tensor(ref_data["name"], ref_data["dtype"])
        if ref_tensor is not None:
            ref_name = os.path.basename(ref_data["name"])

            plot_items.append((f"Ref: {ref_name}", ref_tensor, "spatial"))

            if ref_tensor.shape == target_tensor.shape:
                safe_ref = np.where(ref_tensor == 0, 1.0, ref_tensor)
                diff_tensor = np.abs(ref_tensor - target_tensor) / np.abs(safe_ref)

                diff_tensor[ref_tensor == 0] = 0.0

                plot_items.append(
                    (f"Err Map: vs {ref_name}", diff_tensor, "spatial_log")
                )

                plot_items.append(
                    (f"Err Hist: vs {ref_name}", diff_tensor, "histogram")
                )
            else:
                print(f"Shape mismatch: {ref_name}")

    n_plots = len(plot_items)
    cols = 3
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    if n_plots == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    for i, (title, tensor, p_type) in enumerate(plot_items):
        ax = axes[i]
        try:
            if p_type == "histogram":
                plot_error_histogram(tensor, ax, title)
            else:
                use_log = p_type == "spatial_log"
                plot_tensor(tensor, ax, title, log_scale=use_log)
        except Exception as e:
            print(f"Error plotting {title}: {e}")
            ax.set_title(f"Error: {title}")

    for j in range(len(plot_items), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
