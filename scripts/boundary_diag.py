import argparse
import numpy as np
import pandas as pd
import struct
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

PLOT_DIR = "./data/plots/other"


def fp128_to_float64(raw16: bytes) -> float:
    """Convert 16-byte IEEE 754 quad-precision to Python float (float64)."""
    lo, hi = struct.unpack("<QQ", raw16)
    sign = (hi >> 63) & 1
    exp = (hi >> 48) & 0x7FFF
    frac_hi = hi & 0x0000FFFFFFFFFFFF
    frac_lo = lo
    if exp == 0x7FFF:
        if frac_hi == 0 and frac_lo == 0:
            return float("-inf") if sign else float("inf")
        return float("nan")
    if exp == 0:
        # Subnormal: value = (-1)^sign * 2^(-16382) * (frac/2^112)
        mantissa = (frac_hi * (2**64) + frac_lo) / (2**112)
        return (-1 if sign else 1) * 2 ** (-16382) * mantissa
    # Normal: value = (-1)^sign * 2^(exp-16383) * (1 + frac/2^112)
    mantissa = 1.0 + (frac_hi * (2**64) + frac_lo) / (2**112)
    value = (-1 if sign else 1) * (2 ** (exp - 16383)) * mantissa
    return value


def read_tensor(path):
    with open(path, "rb") as f:
        raw = f.read()
    ndim, type_id = struct.unpack("<QQ", raw[:16])
    shape = struct.unpack(f"<{'Q' * ndim}", raw[16 : 16 + 8 * ndim])
    offset = 16 + 8 * ndim
    n = 1
    for s in shape:
        n *= s

    if type_id == 0:  # float32
        data = np.frombuffer(raw[offset:], dtype=np.float32).copy().astype(np.float64)
    elif type_id == 1:  # float64
        data = np.frombuffer(raw[offset:], dtype=np.float64).copy()
    elif type_id == 2:  # long double (80-bit, stored as 16 bytes on Linux)
        data = (
            np.frombuffer(raw[offset:], dtype=np.longdouble).copy().astype(np.float64)
        )
    elif type_id == 3:  # __float128 (16 bytes, IEEE quad precision)
        data = np.array(
            [
                fp128_to_float64(raw[offset + 16 * i : offset + 16 * i + 16])
                for i in range(n)
            ]
        )
    else:
        raise ValueError(f"Unknown type_id={type_id}")

    return data.reshape(shape)


def make_boundary_mask(shape, stencil_offsets):
    """Returns (interior_mask, boundary_mask) where interior=True."""
    interior = np.ones(shape, dtype=bool)
    for dim, off in enumerate(stencil_offsets):
        slices_lo = [slice(None)] * len(shape)
        slices_hi = [slice(None)] * len(shape)
        slices_lo[dim] = slice(0, off)
        slices_hi[dim] = slice(shape[dim] - off, None)
        interior[tuple(slices_lo)] = False
        interior[tuple(slices_hi)] = False
    return interior, ~interior


def rmse(arr):
    return float(np.sqrt(np.mean(arr**2)))


def analyze(ref_path, oz_path, stencil_offsets, name, n_slices, coeffs):
    ref = read_tensor(ref_path)
    oz = read_tensor(oz_path)
    assert ref.shape == oz.shape, f"Shape mismatch: {ref.shape} vs {oz.shape}"

    interior_mask, boundary_mask = make_boundary_mask(ref.shape, stencil_offsets)
    err = ref - oz
    n_bnd = int(np.sum(boundary_mask))
    n_int = int(np.sum(interior_mask))
    n_tot = err.size

    c = np.array(coeffs)
    return {
        "name": name,
        "n_slices": n_slices,
        "rmse_total": rmse(err),
        "rmse_bnd": rmse(err[boundary_mask]) if n_bnd > 0 else 0.0,
        "rmse_int": rmse(err[interior_mask]) if n_int > 0 else 0.0,
        "bnd_frac": f"{100 * n_bnd / n_tot:.1f}%",
        "n_coeffs": len(c),
        "l1_norm": float(np.sum(np.abs(c))),
        "n_nonzero": int(np.count_nonzero(c)),
        "coeffs": coeffs,
    }


def save_plots(ref_path, oz_path, ndim, name, n_slices):
    """Save histogram and error-tensor plots for one stencil+slice case."""
    ref = read_tensor(ref_path)
    oz = read_tensor(oz_path)
    abs_err = np.abs(ref - oz)
    eps = np.finfo(float).tiny          # floor for log scale
    stem = f"{name}_S{n_slices}"
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # (a) Histogram  (log-spaced bins on x-axis)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    flat = abs_err.ravel()
    pos = flat[flat > 0]
    if pos.size:
        bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 60)
        ax.hist(pos, bins=bins, color="steelblue", edgecolor="none")
        ax.set_xscale("log")
    else:
        ax.hist(flat, bins=60, color="steelblue", edgecolor="none")
        ax.text(0.5, 0.5, "all errors are zero", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
    ax.set_xlabel("|Error|")
    ax.set_ylabel("Count")
    ax.set_title(f"{name}  S={n_slices}  —  Error histogram")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"hist_{stem}.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # (b) Error-tensor plot in logarithmic scale
    # ------------------------------------------------------------------
    log_err = np.log10(np.maximum(abs_err, eps))

    if ndim == 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(abs_err, lw=0.8, color="steelblue")
        if pos.size:
            ax.set_yscale("log")
        ax.set_xlabel("Index")
        ax.set_ylabel("|Error|")
        ax.set_title(f"{name}  S={n_slices}  —  Error (1D, log scale)")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"err_1d_{stem}.png"), dpi=150)
        plt.close(fig)

    elif ndim == 2:
        vmin = float(log_err[abs_err > 0].min()) if pos.size else -16.0
        vmax = float(log_err.max())
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(
            log_err, origin="lower", aspect="auto",
            cmap="hot_r",
            vmin=vmin, vmax=vmax,
        )
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("log₁₀|Error|")
        ax.set_title(f"{name}  S={n_slices}  —  Error heatmap (log scale)")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 0")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"err_2d_{stem}.png"), dpi=150)
        plt.close(fig)

    elif ndim == 3:
        if not HAS_PLOTLY:
            print(f"[WARN] plotly not installed — skipping 3D isosurface for {stem}.")
            return
        shape = abs_err.shape
        # Build coordinate arrays
        xi, yi, zi = np.mgrid[
            0:shape[0], 0:shape[1], 0:shape[2]
        ]
        values = log_err.ravel().astype(float)
        vmin = float(log_err[abs_err > 0].min()) if pos.size else -16.0
        vmax = float(log_err.max())
        fig3d = go.Figure(
            go.Isosurface(
                x=xi.ravel().astype(float),
                y=yi.ravel().astype(float),
                z=zi.ravel().astype(float),
                value=values,
                isomin=vmin,
                isomax=vmax,
                surface_count=6,
                colorscale="Hot",
                reversescale=True,
                opacity=0.35,
                colorbar=dict(title="log₁₀|Error|"),
                caps=dict(x_show=False, y_show=False, z_show=False),
            )
        )
        fig3d.update_layout(
            title=f"{name}  S={n_slices}  —  Error isosurfaces (log₁₀ scale)",
            scene=dict(
                xaxis_title="dim 0",
                yaxis_title="dim 1",
                zaxis_title="dim 2",
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        fig3d.write_html(os.path.join(PLOT_DIR, f"err_3d_{stem}.html"))


TENSOR_DIR = "./data/tensors"
INPUT_LABEL_MAP = {1: "1D_128", 2: "2D_128x128", 3: "3D_32x32x32"}
DEFAULT_PRECISION = "fp128"
DEFAULT_TIMESTEPS = 40


def parse_stencil_file(path):
    """Parse name, shape, offsets, precision, and coeffs from a .stc stencil spec file."""
    name = None
    shape = None
    offsets = None
    precision = "double"
    coeffs = []
    reading_coeffs = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                reading_coeffs = False if not line else reading_coeffs
                continue
            if line.startswith("name "):
                name = line[5:].strip()
                reading_coeffs = False
            elif line.startswith("shape "):
                shape = list(map(int, line[6:].split()))
                reading_coeffs = False
            elif line.startswith("offsets "):
                offsets = list(map(int, line[8:].split()))
                reading_coeffs = False
            elif line.startswith("precision "):
                precision = line[10:].strip()
                reading_coeffs = False
            elif line == "coeffs":
                reading_coeffs = True
            elif reading_coeffs:
                coeffs.extend(map(float, line.split()))
    if shape is None:
        raise ValueError(f"No 'shape' found in {path}")
    if name is None:
        raise ValueError(f"No 'name' found in {path}")
    if offsets is None:
        offsets = [s // 2 for s in shape]
    return name, shape, offsets, precision, coeffs


parser = argparse.ArgumentParser(
    description="Analyze boundary vs interior error of Ozaki stencil results."
)
parser.add_argument(
    "--stencils",
    nargs="+",
    metavar="FILE",
    required=True,
    help="One or more .stc stencil files to analyze.",
)
parser.add_argument(
    "--slices",
    nargs="+",
    type=int,
    default=[8],
    help="One or more INT8 slice counts to analyze. Default: 8.",
)
parser.add_argument(
    "--plot",
    action="store_true",
    help=f"Save error histograms and tensor plots to {PLOT_DIR}/.",
)
args = parser.parse_args()

cases = []
for stc_path in args.stencils:
    name, shape, offsets, precision, coeffs = parse_stencil_file(stc_path)
    ndim = len(shape)
    if ndim not in INPUT_LABEL_MAP:
        print(f"Unsupported dimensionality {ndim} in {stc_path}, skipping.")
        continue
    input_label = INPUT_LABEL_MAP[ndim]
    T = DEFAULT_TIMESTEPS
    ref_name = f"ref_{input_label}_{name}_{DEFAULT_PRECISION}_{T}.dat"
    for s in args.slices:
        oz_name = f"oz_{input_label}_{name}_{T}_{s}.dat"
        cases.append((ref_name, oz_name, offsets, name, s, ndim, coeffs))

rows = []
for ref_name, oz_name, offsets, name, n_slices, ndim, coeffs in cases:
    ref_path = os.path.join(TENSOR_DIR, ref_name)
    oz_path = os.path.join(TENSOR_DIR, oz_name)
    if not os.path.exists(ref_path):
        print(f"[MISSING REF]  {name} S={n_slices}: {ref_name}")
        continue
    if not os.path.exists(oz_path):
        print(f"[MISSING OZ]   {name} S={n_slices}: {oz_name}")
        continue
    try:
        rows.append(analyze(ref_path, oz_path, offsets, name, n_slices, coeffs))
        if args.plot:
            save_plots(ref_path, oz_path, ndim, name, n_slices)
    except Exception as e:
        print(f"[ERROR] {name} S={n_slices}: {e}")

if rows:
    df = pd.DataFrame(rows)
    df = df.set_index(["name", "n_slices"]).sort_index()
    pd.options.display.float_format = "{:.4e}".format
    print(df.drop(columns=["coeffs"]).sort_values("l1_norm").to_string())
