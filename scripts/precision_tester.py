import numpy as np
import subprocess
import glob
import csv
import os
import re
import sys
import argparse
from contextlib import nullcontext
from decimal import Decimal, getcontext
from itertools import product

getcontext().prec = 60

STENCIL_NAME = ["0"]
REQUESTED_SLICES = [1, 2, None]
NEGATIVE_COEFFS = [True, False]
SPARSE = [True, False]
L1_NORM = ["ONE", "GT1", "LT1"]

TIMESTEPS = [1, 10, 25]

EXTRA_LSBS = 0
EXTRA_MSBS = 0

# Entry: (stencil_shape, label). Dimensionality is len(stencil_shape); the
# input tensor for that dim is supplied via a positional CLI arg.
SHAPE_INPUT_PAIRS = [
    ([5], "1D"),
    # ([7], "1D"),
    # ([3, 3], "2D"),
    # ([5, 5], "2D"),
    ([7, 7], "2D"),
    ([3, 3, 3], "3D"),
    # ([5, 5, 5], "3D"),
]

ERROR_THRESHOLD = 1e-8

BUILD_BIN = "./build/bin"
TENSOR_DIR = "./data/tensors"
STENCIL_DIR = "./data/stencils"
CSV_OUTPUT = "./data/precision_results.csv"
BREACHES_OUTPUT = "./data/breaches.txt"


def _pick_target_l1(l1_norm, rng):
    if l1_norm == "ONE":
        return 1.0 / (1.0 + 0.05 * (1.0 - 2 * rng.random()))
    elif l1_norm == "GT1":
        return rng.uniform(1.05, 1.5)
    elif l1_norm == "LT1":
        return rng.uniform(0.3, 0.95)
    else:
        raise ValueError(f"Unknown l1_norm mode: {l1_norm}")


def _format_decimal(d):
    """Plain decimal string (no exponent), trimming trailing zeros after the point."""
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def generate_stencil_spec(
    shape,
    negative_coeffs=True,
    requested_slices=None,
    stencil_num=0,
    offsets=None,
    sparse=False,
    l1_norm="ONE",
    rng=None,
):
    """Generate a stencil spec written in fp128.

    With a basis (requested_slices set): coefficients are computed as exact
    rationals (int_coeff * target_l1 / l1_int) using Decimal, then emitted with
    full 50+ digit precision so that fp128 parsing keeps every bit we care about.

    Without a basis: coefficients are generated as FP64 standard
    normal then rescaled to target L1, and we append the decimal digits of a
    uniformly random 64-bit integer to the string form of each nonzero
    coefficient. Zero stays zero.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if requested_slices:
        abs_limit = 2 ** (7 * requested_slices) // np.prod(shape)
        int_coeffs = rng.integers(
            -abs_limit if negative_coeffs else 0, abs_limit, size=shape
        )

        if sparse:
            zero_mask = rng.random(size=int_coeffs.shape) < 0.7
            int_coeffs[zero_mask] = 0

        l1_int = int(np.sum(np.abs(int_coeffs)))
        if l1_int == 0:
            int_coeffs.flat[len(int_coeffs.flat) // 2] = 1
            l1_int = 1

        target_l1 = _pick_target_l1(l1_norm, rng)

        target_l1_d = Decimal(repr(target_l1))
        l1_int_d = Decimal(int(l1_int))
        coeffs_str = []
        for ic in int_coeffs.flatten():
            ic = int(ic)
            if ic == 0:
                coeffs_str.append("0")
            else:
                val = (Decimal(ic) * target_l1_d) / l1_int_d
                coeffs_str.append(_format_decimal(val))
    else:
        coeffs = rng.standard_normal(size=shape).astype(np.float64)
        if not negative_coeffs:
            coeffs = np.abs(coeffs)

        if sparse:
            zero_mask = rng.random(size=coeffs.shape) < 0.7
            coeffs[zero_mask] = 0.0

        l1_sum = float(np.sum(np.abs(coeffs)))
        if l1_sum == 0:
            coeffs.flat[len(coeffs.flat) // 2] = 1.0
            l1_sum = 1.0

        target_l1 = _pick_target_l1(l1_norm, rng)
        coeffs *= target_l1 / l1_sum

        coeffs_str = []
        for c in coeffs.flatten():
            c = float(c)
            if c == 0.0:
                coeffs_str.append("0")
                continue
            coeffs_str.append(_format_decimal(Decimal(c)))

    if not offsets:
        offsets = np.array(shape) // 2
    shape = [str(x) for x in shape]
    offsets = [str(x) for x in offsets]

    l1_tag = {"ONE": "N", "GT1": "G", "LT1": "L"}[l1_norm]
    name = (
        f"{'x'.join(shape)}"
        f"{'Pos' if not negative_coeffs else ''}"
        f"{'Sp' if sparse else ''}"
        f"{l1_tag}"
        f"m{requested_slices if requested_slices else 'None'}"
        f"_{stencil_num}"
    )

    spec = f"""
    name {name}
    shape {" ".join(shape)}
    offsets {" ".join(offsets)}
    precision fp128
    coeffs
    {" ".join(coeffs_str)}
    """
    with open(f"{STENCIL_DIR}/{name}.stc", "w") as outfile:
        outfile.write(spec)
    return name


def run_pipeline(name, input_file, input_label, T, algorithm="direct"):
    stencil_file = f"{STENCIL_DIR}/{name}.stc"
    ref_prefix = f"ref_{input_label}"
    oz_prefix = f"oz_{input_label}"
    ref_file = f"{TENSOR_DIR}/{ref_prefix}_{name}_fp128_{T}.dat"

    # Step 1: compute-reference
    ref_result = subprocess.run(
        [
            f"{BUILD_BIN}/compute-reference",
            "-i",
            input_file,
            "--stencil-file",
            stencil_file,
            "--prefix",
            ref_prefix,
            "-T",
            str(T),
        ],
        capture_output=True,
        text=True,
    )
    if ref_result.returncode != 0:
        print(f"  [FAIL] compute-reference for {name} | {input_label} | T={T}:")
        print(
            f"    {ref_result.stderr.splitlines()[-1] if ref_result.stderr else '(no output)'}"
        )
        return False, [float("nan")], ""

    # Step 2: run-ozaki with --save-tensors to write output files to disk
    oz_result = subprocess.run(
        [
            f"{BUILD_BIN}/run-ozaki",
            "-i",
            input_file,
            "--stencil-file",
            stencil_file,
            "--prefix",
            oz_prefix,
            "-T",
            str(T),
            "--extra-lsb",
            str(EXTRA_LSBS),
            "--extra-msb",
            str(EXTRA_MSBS),
            "--save-tensors",
            "--algorithm",
            algorithm,
        ],
        capture_output=True,
        text=True,
    )
    oz_output = oz_result.stdout + oz_result.stderr
    if oz_result.returncode != 0:
        print(f"  [FAIL] run-ozaki for {name} | {input_label} | T={T}:")
        print(
            f"    {oz_result.stderr.splitlines()[-1] if oz_result.stderr else '(no output)'}"
        )
        return False, [float("nan")], oz_output

    # Step 3: find the highest-slice output file written by run-ozaki
    oz_pattern = f"{TENSOR_DIR}/{oz_prefix}_{name}_{T}_*.dat"
    oz_files = glob.glob(oz_pattern)
    if not oz_files:
        print(f"  [FAIL] No ozaki output files found matching: {oz_pattern}")
        return False, [float("nan")], oz_output

    max_slices = max(
        int(os.path.basename(f).rsplit("_", 1)[-1].replace(".dat", ""))
        for f in oz_files
    )
    errs = []
    for i in range(2, max_slices):
        oz_file = f"{TENSOR_DIR}/{oz_prefix}_{name}_{T}_{i}.dat"

        # Step 4: compute-errors, RMSE is printed to stdout
        err_result = subprocess.run(
            [f"{BUILD_BIN}/compute-errors", "-e", ref_file, "-a", oz_file, "--scaled"],
            capture_output=True,
            text=True,
        )
        if err_result.returncode != 0:
            print(f"  [FAIL] compute-errors for {name} | {input_label} | T={T}:")
            print(
                f"    {err_result.stderr.splitlines()[-1] if err_result.stderr else '(no output)'}"
            )
            return False, [float("nan")], oz_output

        try:
            error = float(err_result.stdout.strip())
            errs.append(error)
        except ValueError:
            print(
                f"  [FAIL] compute-errors unparseable output: '{err_result.stdout.strip()}'"
            )
            return False, [float("nan")], oz_output

    return True, errs, oz_output


def _decode_stencil_name(name):
    """Parse metadata encoded in a stencil name string.

    Name format: {shape}[Pos][Sp]{G|L|N}m{req_slices}_{stencil_num}
    Returns (shape_str, has_negative, is_sparse, l1_tag, req_slices_str, stencil_num).
    """
    # Try new format first (with L1 tag, optional Sp)
    m = re.match(r"^([0-9x]+)(Pos)?(Sp)?([GLN])m(.+)_(\d+)$", name)
    if m:
        shape_str = m.group(1)
        has_negative = m.group(2) is None
        is_sparse = m.group(3) is not None
        l1_tag = m.group(4)
        req_slices_str = m.group(5)
        return (
            shape_str,
            has_negative,
            is_sparse,
            l1_tag,
            req_slices_str,
            int(m.group(6)),
        )

    # Fallback: legacy format without L1 tag / sparse marker
    m = re.match(r"^([0-9x]+)(Pos)?m(.+)_(\d+)$", name)
    if m:
        shape_str = m.group(1)
        has_negative = m.group(2) is None
        req_slices_str = m.group(3)
        return shape_str, has_negative, False, None, req_slices_str, int(m.group(4))

    return None, None, None, None, None, None


def plot_results(mode):
    """Read precision_results.csv and produce a grid of RMSE-vs-slice-count plots.
    Columns : one per unique stencil shape, in SHAPE_INPUT_PAIRS order.
    Rows    : determined by *mode*:
              "pos-neg" 2 rows (negative-coefficient stencils / positive-only)
              "slices"  len(REQUESTED_SLICES) rows, one per requested-slice value
              "sparse"  2 rows (sparse / dense)
              "L1"      3 rows (ONE / GT1 / LT1)
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    if not os.path.exists(CSV_OUTPUT):
        print(f"CSV file not found: {CSV_OUTPUT}. Run without --plot first.")
        sys.exit(1)

    df = pd.read_csv(CSV_OUTPUT)
    if df.empty:
        print("CSV is empty, nothing to plot.")
        return

    # Decode stencil metadata from the name column.
    decoded = df["stencil"].apply(_decode_stencil_name)
    df["stencil_shape"] = [d[0] for d in decoded]
    df["has_negative"] = [d[1] for d in decoded]
    df["is_sparse"] = [d[2] for d in decoded]
    df["l1_tag"] = [d[3] for d in decoded]
    df["req_slices"] = [d[4] for d in decoded]  # str: "1","2","3","None"

    # Keep only rows that have numeric per-slice data.
    df = df[df["status"] == "Completed"].copy()
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    df["n_slices"] = pd.to_numeric(df["n_slices"], errors="coerce")
    df = df.dropna(subset=["rmse", "n_slices", "stencil_shape"])
    df["n_slices"] = df["n_slices"].astype(int)

    # Column order: unique shapes in the order they appear in SHAPE_INPUT_PAIRS.
    shape_order = []
    for stencil_shape, _ in SHAPE_INPUT_PAIRS:
        s = "x".join(str(x) for x in stencil_shape)
        if s not in shape_order:
            shape_order.append(s)

    # Row definitions differ by mode.
    if mode == "pos-neg":
        row_vals = [True, False]
        row_labels = ["Negative coefficients", "Positive only"]
        row_col = "has_negative"
    elif mode == "slices":
        row_vals = [str(v) if v is not None else "None" for v in REQUESTED_SLICES]
        row_labels = [
            f"Req. slices: {v if v is not None else 'None'}" for v in REQUESTED_SLICES
        ]
        row_col = "req_slices"
    elif mode == "sparse":
        row_vals = [True, False]
        row_labels = ["Sparse", "Dense"]
        row_col = "is_sparse"
    elif mode == "L1":
        row_vals = ["N", "G", "L"]
        row_labels = ["L1 ~ 1 (stable)", "L1 > 1 (expansive)", "L1 < 1 (contractive)"]
        row_col = "l1_tag"

    n_rows = len(row_vals)
    n_cols = len(shape_order)

    timesteps = sorted(df["T"].unique())
    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    t_colors = {T: cycle_colors[i % len(cycle_colors)] for i, T in enumerate(timesteps)}

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    group_desc = {
        "pos-neg": "coefficient sign",
        "slices": "requested slice count",
        "sparse": "sparsity",
        "L1": "L1 norm magnitude",
    }[mode]
    fig.suptitle(
        f"Ozaki stencil precision grouped by {group_desc}",
        fontsize=13,
        y=1.01,
    )

    # Collect one Line2D handle per T for the shared figure legend.
    legend_handles = {}

    for r_idx, (r_val, r_label) in enumerate(zip(row_vals, row_labels)):
        for c_idx, shape in enumerate(shape_order):
            ax = axes[r_idx, c_idx]

            cell_df = df[(df["stencil_shape"] == shape) & (df[row_col] == r_val)]

            has_data = False
            for T in timesteps:
                t_df = cell_df[cell_df["T"] == T]
                if t_df.empty:
                    continue

                # Aggregate over all stencil instances sharing (shape, row, T, n_slices).
                agg = (
                    t_df.groupby("n_slices")["rmse"]
                    .agg(mean="mean", lo="min", hi="max")
                    .reset_index()
                    .sort_values("n_slices")
                )
                if agg.empty:
                    continue

                color = t_colors[T]
                (line,) = ax.plot(
                    agg["n_slices"],
                    agg["mean"],
                    color=color,
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                    label=f"T={T}",
                )
                ax.fill_between(
                    agg["n_slices"],
                    agg["lo"],
                    agg["hi"],
                    color=color,
                    alpha=0.15,
                )
                legend_handles[T] = line
                has_data = True

            if not has_data:
                ax.set_visible(False)
                continue

            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
            ax.tick_params(labelsize=7)
            ax.set_xlabel("Slice count", fontsize=8)
            ax.set_title(f"Shape {shape}", fontsize=9, fontweight="bold")
            # Left-column axes carry the row label as part of the y-axis title.
            if c_idx == 0:
                ax.set_ylabel(f"{r_label}\nRMSE", fontsize=8)
            else:
                ax.set_ylabel("RMSE", fontsize=8)

    # Single figure-level legend for timestep lines, centred below the grid.
    if legend_handles:
        sorted_T = sorted(legend_handles)
        fig.legend(
            [legend_handles[T] for T in sorted_T],
            [f"T = {T}" for T in sorted_T],
            loc="lower center",
            ncol=len(sorted_T),
            fontsize=9,
            bbox_to_anchor=(0.5, -0.03),
            frameon=True,
        )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = f"precision_{mode}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")
    plt.show()


_parser = argparse.ArgumentParser(
    description="Ozaki stencil precision tester and visualiser."
)
_parser.add_argument(
    "--algorithm",
    "-A",
    choices=["direct", "convstencil"],
    default="direct",
    help=(
        "Algorithm passed to run-ozaki: "
        "'direct' (scalar stencil_v1, default) or "
        "'convstencil' (WMMA tensor core, 2D 7x7 stencils only)."
    ),
)
_parser.add_argument(
    "--no-experiments",
    action="store_true",
    help="Generate stencil files only; skip running the precision experiments.",
)
_parser.add_argument(
    "--plot",
    choices=["pos-neg", "slices", "sparse", "L1"],
    metavar="{pos-neg,slices,sparse,L1}",
    help=(
        "Read existing results from the CSV and produce a "
        "precision plot without re-running the test pipeline.  "
        "'pos-neg' groups rows by coefficient sign; "
        "'slices' groups rows by requested slice count; "
        "'sparse' groups rows by sparsity (sparse vs dense); "
        "'L1' groups rows by L1 norm magnitude (stable / expansive / contractive)."
    ),
)
_parser.add_argument(
    "tensor_1d",
    help="Path to 1D input tensor, or NONE to skip 1D tests.",
    nargs="?",
    default=None,
)
_parser.add_argument(
    "tensor_2d",
    help="Path to 2D input tensor, or NONE to skip 2D tests.",
    nargs="?",
    default=None,
)
_parser.add_argument(
    "tensor_3d",
    help="Path to 3D input tensor, or NONE to skip 3D tests.",
    nargs="?",
    default=None,
)
_args = _parser.parse_args()

if _args.plot:
    plot_results(_args.plot)
    sys.exit(0)


# --- Setup ---
os.makedirs(STENCIL_DIR, exist_ok=True)
os.makedirs(TENSOR_DIR, exist_ok=True)

rng = np.random.default_rng(0)


def _resolve_tensor_arg(arg):
    if arg is None or arg.upper() == "NONE":
        return None
    if not os.path.isfile(arg):
        raise FileNotFoundError(f"Input tensor not found: {arg}")
    return arg


input_tensors = {
    1: _resolve_tensor_arg(_args.tensor_1d),
    2: _resolve_tensor_arg(_args.tensor_2d),
    3: _resolve_tensor_arg(_args.tensor_3d),
}
for dim, path in input_tensors.items():
    if path is None:
        if not _args.no_experiments:
            print(f"Skipping {dim}D tests (no input tensor supplied).")
    else:
        print(f"Using {dim}D input tensor: {path}")

# --- Precision test loop ---
_csv_ctx = (
    nullcontext(None) if _args.no_experiments else open(CSV_OUTPUT, "w", newline="")
)
_breach_ctx = (
    nullcontext(None)
    if _args.no_experiments
    else open(BREACHES_OUTPUT, "w", newline="")
)
with _csv_ctx as csvfile, _breach_ctx as breachfile:
    if not _args.no_experiments:
        writer = csv.writer(csvfile)
        # One row per (stencil, input_shape, T, n_slices) so the plotter can draw
        # RMSE as a function of slice count.
        writer.writerow(
            [
                "stencil",
                "input_shape",
                "T",
                "extra_in_msbs",
                "extra_in_lsbs",
                "msb_shift_stencil",
                "status",
                "result",
                "n_slices",
                "rmse",
            ]
        )

    for stencil_shape, input_label in SHAPE_INPUT_PAIRS:
        dim = len(stencil_shape)
        input_file = input_tensors.get(dim)
        if input_file is None and not _args.no_experiments:
            continue
        for neg_coeffs, req_slices, sparse, l1_norm, stencil_num in product(
            NEGATIVE_COEFFS, REQUESTED_SLICES, SPARSE, L1_NORM, STENCIL_NAME
        ):
            name = generate_stencil_spec(
                shape=stencil_shape,
                negative_coeffs=neg_coeffs,
                requested_slices=req_slices,
                stencil_num=stencil_num,
                sparse=sparse,
                l1_norm=l1_norm,
                rng=rng,
            )
            if _args.no_experiments:
                print(f"Generated stencil: {name}")
                continue
            for T in TIMESTEPS:
                print(f"Testing {name} | {input_label} | T={T} ...")
                success, errs, oz_output = run_pipeline(
                    name, input_file, input_label, T, _args.algorithm
                )
                stencil_msb_shift = oz_output.split("\n")[0][-1] if oz_output else ""

                if success:
                    high_error = errs[-1] > ERROR_THRESHOLD
                    # Write one CSV row per slice count for plotting.
                    for i, err in enumerate(errs):
                        n_slices = i + 2
                        writer.writerow(
                            [
                                name,
                                input_label,
                                T,
                                EXTRA_MSBS,
                                EXTRA_LSBS,
                                stencil_msb_shift,
                                "Completed",
                                "Wrong" if err > ERROR_THRESHOLD else "Correct",
                                n_slices,
                                err,
                            ]
                        )
                    print(f"  [OK] error = {errs[-1]:.6e}")
                    if high_error:
                        print(
                            f"  [WARN] error exceeds threshold {ERROR_THRESHOLD:.0e}."
                        )
                        breachfile.write(f"{name}\n")
                        breachfile.write(f"{T=}\n")
                        breachfile.write(
                            ",\n".join(
                                [f"\t(S={i + 2} : Err={e})" for i, e in enumerate(errs)]
                            )
                            + "\n"
                        )
                        breachfile.write(f"run-ozaki output: {oz_output}\n")
                        breachfile.write(("-" * 90) + "\n\n")
                else:
                    writer.writerow(
                        [
                            name,
                            input_label,
                            T,
                            EXTRA_MSBS,
                            EXTRA_LSBS,
                            stencil_msb_shift,
                            "Failed",
                            "N/A",
                            "",
                            "",
                        ]
                    )

                csvfile.flush()
