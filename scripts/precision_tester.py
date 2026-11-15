import numpy as np
import subprocess
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

# Entry: (stencil_shape, label). Dimensionality is len(stencil_shape); the
# input tensor is generated per stencil halo (see input_tensor).
SHAPE_INPUT_PAIRS = [
    ([3, 3], "2D"),
    ([7, 7], "2D"),
    ([15, 15], "2D"),
    ([3, 3, 3], "3D"),
    ([5, 5, 5], "3D"),
]

_compute_capability = os.environ.get("CUDA_COMPUTE_CAPABILITY")
if _compute_capability is None:
    raise SystemExit("CUDA_COMPUTE_CAPABILITY must be set (for example, 120).")

_compute_capability_match = re.fullmatch(r"([0-9]+)[A-Za-z]*", _compute_capability)
if _compute_capability_match is None:
    raise SystemExit(
        "CUDA_COMPUTE_CAPABILITY must contain a numeric SM followed by an optional "
        f"architecture suffix; got {_compute_capability!r}."
    )

COMPUTE_CAPABILITY = int(_compute_capability_match.group(1))
if COMPUTE_CAPABILITY < 100:
    MAX_SLICES = 12
    ERROR_THRESHOLD = 1.0e-14
    REFERENCE_PRECISION = "double"
else:
    MAX_SLICES = 18
    ERROR_THRESHOLD = 1.0e-31
    REFERENCE_PRECISION = "fp128"

UNCONSTRAINED_SLICES = 9
MAX_COEFFICIENT_DYNAMIC_BITS = 16
MIN_NORMALIZED_TRANSPORT = 0.1

BUILD_BIN = "./build/bin"
TENSOR_DIR = "./data/tensors"
STENCIL_DIR = "./data/stencils"
CSV_OUTPUT = "./data/results/precision_results.csv"
BREACHES_OUTPUT = "./data/results/breaches.txt"


def _format_decimal(d):
    """Plain decimal string (no exponent), trimming trailing zeros after the point."""
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def _cpu_slice_count(values, max_slices=9):
    """Mirror the C++ round-to-nearest base-128 coefficient slicer for FP64."""
    residuals = [float(value) for value in values]
    abs_max = max((abs(value) for value in residuals), default=0.0)
    max_exponent = np.frexp(abs_max)[1] if abs_max else 0
    r_h = 7 * int(np.ceil(max_exponent / 7.0))
    for slice_index in range(max_slices):
        cast_scale = 7 * (slice_index + 1) - r_h
        for index, residual in enumerate(residuals):
            digit = int(np.rint(np.ldexp(residual, cast_scale)))
            digit = max(-128, min(127, digit))
            residuals[index] = residual - np.ldexp(float(digit), -cast_scale)
        if all(value == 0.0 for value in residuals):
            return slice_index + 1
    return max_slices + 1


def _center_flat_index(shape):
    center = tuple(extent // 2 for extent in shape)
    return int(np.ravel_multi_index(center, shape))


def _positions_including_center(shape, count, rng):
    """Choose distinct flat positions, always putting the center first."""
    coefficient_count = int(np.prod(shape))
    if count < 1 or count > coefficient_count:
        raise ValueError("Active coefficient count is outside the stencil size")
    center = _center_flat_index(shape)
    if count == 1:
        return np.array([center], dtype=np.int64)
    candidates = np.delete(np.arange(coefficient_count, dtype=np.int64), center)
    remainder = rng.choice(candidates, size=count - 1, replace=False)
    return np.concatenate((np.array([center], dtype=np.int64), remainder))


def _mirrored_pairs(shape):
    """Return one flat-index pair for every pair of offsets about the center."""
    center = tuple(extent // 2 for extent in shape)
    pairs = []
    for index in np.ndindex(*shape):
        mirror = tuple(2 * middle - coordinate for middle, coordinate in zip(center, index))
        flat = int(np.ravel_multi_index(index, shape))
        mirror_flat = int(np.ravel_multi_index(mirror, shape))
        if flat < mirror_flat:
            pairs.append((flat, mirror_flat))
    return pairs


def _balanced_unconstrained_coefficients(shape, negative_coeffs, sparse, l1_norm, rng):
    """Build a nine-slice stencil without tiny standalone coefficients.

    At least half of the L1 norm is assigned to the center, so information
    cannot leave the domain solely through directed transport. Dense stencils
    use mirrored pairs as a non-transport control. Sparse stencils deliberately
    make one mirrored pair asymmetric: one side carries a controlled transport
    term and the other carries the ninth-slice marker.
    """
    target_l1 = {"LT1": 0.5, "ONE": 1.0, "GT1": 1.5}[l1_norm]
    pairs = _mirrored_pairs(shape)
    if sparse:
        selected_pair = pairs[int(rng.integers(len(pairs)))]
        if rng.integers(2) != 0:
            selected_pair = selected_pair[::-1]

    marker_magnitude = target_l1 / 1024.0
    if sparse:
        center_magnitude = 0.75 * target_l1
        pair_magnitudes = np.array(
            [marker_magnitude, 0.25 * target_l1 - marker_magnitude],
            dtype=np.float64,
        )
    else:
        center_magnitude = 0.5 * target_l1
        pair_magnitudes = np.empty(len(pairs), dtype=np.float64)
        pair_magnitudes[0] = marker_magnitude
        pair_weights = rng.uniform(0.75, 1.25, size=len(pairs) - 1)
        remaining_pair_magnitude = 0.25 * target_l1 - marker_magnitude
        pair_magnitudes[1:] = remaining_pair_magnitude * pair_weights / np.sum(pair_weights)

    # Keep every coefficient on the nine-slice lattice. The marker magnitude
    # is about 2^-10 of the L1 norm, so setting its low lattice bit requires
    # precision without creating a coefficient-scale dynamic-range stress.
    max_exponent = np.frexp(center_magnitude)[1]
    base_exponent = 7 * int(np.ceil(max_exponent / 7.0))
    lowest_quantum = np.ldexp(1.0, base_exponent - 7 * UNCONSTRAINED_SLICES)
    center_magnitude = np.rint(center_magnitude / lowest_quantum) * lowest_quantum
    pair_magnitudes = np.rint(pair_magnitudes / lowest_quantum) * lowest_quantum
    marked = pair_magnitudes[0] + lowest_quantum
    if marked == pair_magnitudes[0]:
        marked = np.nextafter(pair_magnitudes[0], np.inf)
    pair_magnitudes[0] = marked

    coeffs = np.zeros(shape, dtype=np.float64)
    center = _center_flat_index(shape)
    coeffs.flat[center] = center_magnitude
    if sparse:
        marker_position, transport_position = selected_pair
        coeffs.flat[marker_position] = pair_magnitudes[0]
        coeffs.flat[transport_position] = pair_magnitudes[1]
        if negative_coeffs:
            coeffs.flat[center] *= -1.0
    else:
        pair_signs = np.ones(len(pairs), dtype=np.float64)
        if negative_coeffs:
            coeffs.flat[center] *= -1.0
            pair_signs = rng.choice(np.array([-1.0, 1.0]), size=len(pairs))
        for (left, right), magnitude, sign in zip(pairs, pair_magnitudes, pair_signs):
            coeffs.flat[left] = sign * magnitude
            coeffs.flat[right] = sign * magnitude
    return coeffs


def _validate_correctness_coefficients(values, shape, sparse, require_transport=False):
    """Reject stress-test kernels that are unsuitable for correctness tests."""
    coeffs = np.asarray(values, dtype=np.float64).reshape(shape)
    magnitudes = np.abs(coeffs)
    nonzero = magnitudes[magnitudes != 0.0]
    if nonzero.size == 0:
        raise ValueError("Generated stencil has no nonzero coefficients")

    dynamic_bits = float(np.log2(np.max(nonzero) / np.min(nonzero)))
    if dynamic_bits > MAX_COEFFICIENT_DYNAMIC_BITS:
        raise ValueError(
            f"Generated coefficient range spans {dynamic_bits:.1f} bits; "
            f"correctness stencils are limited to {MAX_COEFFICIENT_DYNAMIC_BITS}"
        )

    center_magnitude = magnitudes.flat[_center_flat_index(shape)]
    if center_magnitude == 0.0:
        raise ValueError("Correctness stencils must have a nonzero center coefficient")
    if sparse and center_magnitude < 0.2 * np.sum(magnitudes):
        raise ValueError("Sparse correctness stencils require a persistent center term")

    if require_transport:
        center = np.asarray(shape, dtype=np.float64) // 2
        moment = np.zeros(len(shape), dtype=np.float64)
        for index in np.ndindex(*shape):
            moment += magnitudes[index] * (np.asarray(index) - center)
        normalized_transport = float(np.linalg.norm(moment, ord=1) / np.sum(magnitudes))
        if normalized_transport < MIN_NORMALIZED_TRANSPORT:
            raise ValueError(
                f"Transport stencil has normalized drift {normalized_transport:.3f}; "
                f"expected at least {MIN_NORMALIZED_TRANSPORT}"
            )


def generate_stencil_spec(
    shape,
    negative_coeffs=True,
    requested_slices=None,
    stencil_num=0,
    offsets=None,
    sparse=False,
    l1_norm="ONE",
    rng=None,
    _attempt=0,
):
    """Generate a binary64 stencil specification for precision testing.

    With requested_slices set, coefficients are assembled directly from one or
    two signed base-128 digits and scaled by an integral power of 128. Their L1
    norms are exactly 0.5, 1.0, or 1.5 for the requested category.

    Without requested_slices, coefficients use a balanced, centrosymmetric
    construction on the UNCONSTRAINED_SLICES lattice. This exercises all nine
    coefficient slices without creating an extreme spatial dynamic range.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if requested_slices:
        if requested_slices not in (1, 2):
            raise ValueError("The precision suite supports requested slice counts one and two")

        # At exponent -7*requested_slices, a total high-digit magnitude of
        # 64/128/192 gives L1 0.5/1.0/1.5. Sparse cases use the minimum number
        # of coefficients needed by the signed INT8 digit range; dense cases
        # spread the same magnitude over as many coefficients as possible.
        target_units = {"LT1": 64, "ONE": 128, "GT1": 192}[l1_norm]
        coefficient_count = int(np.prod(shape))
        minimum_active = int(np.ceil(target_units / 63.0))
        active_count = minimum_active if sparse else min(coefficient_count, target_units)
        if active_count > coefficient_count:
            raise ValueError("Stencil is too small for the requested exact digit construction")

        positions = _positions_including_center(shape, active_count, rng)
        magnitudes = np.full(active_count, target_units // active_count, dtype=np.int64)
        magnitudes[: target_units % active_count] += 1
        if np.max(magnitudes) > 63:
            raise AssertionError("Generated high coefficient digit exceeds the conservative INT8 range")

        digits = np.zeros((requested_slices, *shape), dtype=np.int64)
        high = digits[0].reshape(-1)
        high[positions] = magnitudes
        if requested_slices == 2:
            # Opposite low digits keep the requested L1 norm exact while
            # guaranteeing that the lowest numerical slice is required.
            low = digits[1].reshape(-1)
            low[positions[0]] = 1
            low[positions[-1]] = -1

        if negative_coeffs:
            signs = rng.choice(np.array([-1, 1], dtype=np.int64), size=active_count)
            signs[0] = -1
            for slice_index in range(requested_slices):
                plane = digits[slice_index].reshape(-1)
                plane[positions] *= signs

        integer_coeffs = np.zeros(shape, dtype=object)
        for slice_index in range(requested_slices):
            integer_coeffs += digits[slice_index].astype(object) * (
                128 ** (requested_slices - slice_index - 1)
            )
        exponent = -7 * requested_slices
        coeffs_str = []
        for integer in integer_coeffs.flatten():
            integer = int(integer)
            if integer == 0:
                coeffs_str.append("0")
            else:
                coeffs_str.append(_format_decimal(Decimal(integer) * (Decimal(2) ** exponent)))
    else:
        coeffs = _balanced_unconstrained_coefficients(
            shape, negative_coeffs, sparse, l1_norm, rng
        )

        coeffs_str = []
        for c in coeffs.flatten():
            c = float(c)
            if c == 0.0:
                coeffs_str.append("0")
                continue
            coeffs_str.append(_format_decimal(Decimal(c)))

    numeric_coeffs = [float(value) for value in coeffs_str]
    _validate_correctness_coefficients(
        numeric_coeffs,
        shape,
        sparse,
        require_transport=requested_slices is None and sparse,
    )

    if offsets is None:
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

    expected_slices = requested_slices or UNCONSTRAINED_SLICES
    actual_slices = _cpu_slice_count(numeric_coeffs)
    if actual_slices != expected_slices:
        if _attempt >= 100:
            raise RuntimeError(
                f"Could not generate an exact {expected_slices}-slice stencil for shape {shape}"
            )
        return generate_stencil_spec(
            [int(value) for value in shape], negative_coeffs, requested_slices,
            stencil_num, [int(value) for value in offsets], sparse, l1_norm, rng,
            _attempt + 1,
        )

    spec = f"""
    name {name}
    shape {" ".join(shape)}
    offsets {" ".join(offsets)}
    precision double
    coeffs
    {" ".join(coeffs_str)}
    """
    with open(f"{STENCIL_DIR}/{name}.stc", "w") as outfile:
        outfile.write(spec)
    return name


def run_pipeline(name, input_file, input_label, T):
    stencil_file = f"{STENCIL_DIR}/{name}.stc"
    # Names carry the input tensor, so results computed from another input
    # (e.g. a different zero-boundary width) are never reused.
    input_tag = os.path.splitext(os.path.basename(input_file))[0]
    ref_prefix = f"ref_{input_tag}"
    oz_prefix = f"oz_{input_tag}"
    ref_file = (
        f"{TENSOR_DIR}/{ref_prefix}_{name}_{REFERENCE_PRECISION}_{T}.dat"
    )

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
            f"--{REFERENCE_PRECISION}",
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

    # Step 2: run-ozaki writes one output tensor per slice count
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
            "--slice-limit",
            str(MAX_SLICES),
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

    # Step 3: require the configured terminal slice. Looking at the largest
    # glob match could accidentally reuse stale files from another architecture.
    terminal_oz_file = f"{TENSOR_DIR}/{oz_prefix}_{name}_{T}_{MAX_SLICES}.dat"
    if not os.path.isfile(terminal_oz_file):
        print(f"  [FAIL] Missing terminal Ozaki output: {terminal_oz_file}")
        return False, [float("nan")], oz_output

    errs = []
    for i in range(2, MAX_SLICES + 1):
        oz_file = f"{TENSOR_DIR}/{oz_prefix}_{name}_{T}_{i}.dat"

        # Step 4: compute-errors prints RMSE and max absolute difference to stdout
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
            rmse, max_abs_diff = (
                float(value) for value in err_result.stdout.strip().split(",")
            )
            errs.append((rmse, max_abs_diff))
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
    "--dims",
    type=int,
    nargs="+",
    choices=[2, 3],
    default=[2, 3],
    help="Stencil dimensionalities to test (default: 2 3).",
)
_args = _parser.parse_args()

if _args.plot:
    plot_results(_args.plot)
    sys.exit(0)


# --- Setup ---
os.makedirs(STENCIL_DIR, exist_ok=True)
os.makedirs(TENSOR_DIR, exist_ok=True)

rng = np.random.default_rng(0)
rmse_breach_found = False
pipeline_failures = 0


INPUT_EXTENT = {2: 512, 3: 64}


def input_tensor(stencil_shape):
    """Gaussian input whose zero boundary is exactly the stencil's halo.

    A wider zero boundary would make interior cells artificially zero and bias
    the errors in our favour. All variants share the seed (and interior values).
    """
    dim = len(stencil_shape)
    halo = stencil_shape[0] // 2
    n = INPUT_EXTENT[dim]
    path = f"./data/{dim}D_{'x'.join([str(n)] * dim)}_halo{halo}.dat"
    if not os.path.isfile(path):
        print(f"Generating input tensor {path}")
        subprocess.run(
            [f"{BUILD_BIN}/gen-tensor", "-o", path, "--N", str(n),
             "--dims", str(dim), "--halo", str(halo)],
            check=True,
        )
    return path

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
                "max_abs_diff",
            ]
        )

    for stencil_shape, input_label in SHAPE_INPUT_PAIRS:
        if len(stencil_shape) not in _args.dims:
            continue
        input_file = None if _args.no_experiments else input_tensor(stencil_shape)
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
                success, errs, oz_output = run_pipeline(name, input_file, input_label, T)
                stencil_msb_shift = ""

                if success:
                    # The correctness criterion applies only to the terminal
                    # (highest-precision) slice count. Lower-slice results are
                    # retained for convergence plots, but are informational.
                    final_rmse = errs[-1][0]
                    high_error = final_rmse > ERROR_THRESHOLD
                    rmse_breach_found = rmse_breach_found or high_error
                    # Write one CSV row per slice count for plotting.
                    for i, (rmse, max_abs_diff) in enumerate(errs):
                        n_slices = i + 2
                        if i == len(errs) - 1:
                            result = "Wrong" if high_error else "Correct"
                        else:
                            result = "Informational"
                        writer.writerow(
                            [
                                name,
                                input_label,
                                T,
                                0,
                                0,
                                stencil_msb_shift,
                                "Completed",
                                result,
                                n_slices,
                                rmse,
                                max_abs_diff,
                            ]
                        )
                    print(f"  [OK] error = {errs[-1][0]:.6e}")
                    if high_error:
                        print(
                            f"  [WARN] error exceeds threshold {ERROR_THRESHOLD:.0e}."
                        )
                        breachfile.write(f"{name}\n")
                        breachfile.write(f"{T=}\n")
                        breachfile.write(
                            ",\n".join(
                                [
                                    f"\t(S={i + 2} : RMSE={rmse}, MaxAbsDiff={max_abs_diff})"
                                    for i, (rmse, max_abs_diff) in enumerate(errs)
                                ]
                            )
                            + "\n"
                        )
                        breachfile.write(f"run-ozaki output: {oz_output}\n")
                        breachfile.write(("-" * 90) + "\n\n")
                else:
                    pipeline_failures += 1
                    writer.writerow(
                        [
                            name,
                            input_label,
                            T,
                            0,
                            0,
                            stencil_msb_shift,
                            "Failed",
                            "N/A",
                            "",
                            "",
                            "",
                        ]
                    )

                csvfile.flush()

if not _args.no_experiments:
    if not rmse_breach_found and pipeline_failures == 0:
        print(
            f"\nSUCCESS: no stencil produced an RMSE above the "
            f"{ERROR_THRESHOLD:.0e} threshold."
        )
    elif pipeline_failures:
        print(
            f"\nPrecision testing completed with {pipeline_failures} pipeline failure(s); "
            "no success claim was made."
        )
