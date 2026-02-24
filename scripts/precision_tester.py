import numpy as np
import subprocess
import struct
import glob
import csv
import os
from itertools import product

STENCIL_NAME = ["0", "1", "2", "3"]
REQUESTED_SLICES = [1, 2, 3, None]
NEGATIVE_COEFFS = [True, False]

TIMESTEPS = [1, 10, 100]

# If True, input tensors will have zeros on the boundary (Dirichlet boundary condition)
DIRICHLET_ZERO = True


EXTRA_LSBS = 8
EXTRA_MSBS = 0

# Each entry: (stencil_shape, input_shape, label)
# Stencil dimensionality MUST match input dimensionality.
SHAPE_INPUT_PAIRS = [
    ([5], [128], "1D_128"),
    ([7], [128], "1D_128"),
    ([3, 3], [128, 128], "2D_128x128"),
    ([5, 5], [128, 128], "2D_128x128"),
    ([7, 7], [128, 128], "2D_128x128"),
    ([3, 3, 3], [32, 32, 32], "3D_32x32x32"),
    ([5, 5, 5], [32, 32, 32], "3D_32x32x32"),
]

ERROR_THRESHOLD = 1e-8

BUILD_BIN = "./build/bin"
DATA_DIR = "./data"
TENSOR_DIR = "./data/tensors"
STENCIL_DIR = "./data/stencils"
CSV_OUTPUT = "precision_results.csv"
BREACHES_OUTPUT = "breaches.txt"

PRECISION_TYPE_MAP = {
    "double": np.float64,
    "float": np.float32,
    "longdouble": np.longdouble,
    "fp128": np.float128 if hasattr(np, "float128") else np.longdouble,
}


def apply_dirichlet_boundary(data, stencil_shape):
    data = np.asarray(data, dtype=np.float64)
    ndim = len(data.shape)
    stencil_shape = list(stencil_shape)

    if ndim != len(stencil_shape):
        raise ValueError(
            f"Data dimensionality {ndim} != stencil dimensionality {len(stencil_shape)}"
        )

    # Calculate boundary width for each dimension (offset = stencil_shape // 2)
    boundary_widths = [s // 2 for s in stencil_shape]

    if ndim == 1:
        bw = boundary_widths[0]
        data[:bw] = 0.0
        data[-bw:] = 0.0
    elif ndim == 2:
        bw0, bw1 = boundary_widths
        data[:bw0, :] = 0.0
        data[-bw0:, :] = 0.0
        data[:, :bw1] = 0.0
        data[:, -bw1:] = 0.0
    elif ndim == 3:
        bw0, bw1, bw2 = boundary_widths
        data[:bw0, :, :] = 0.0
        data[-bw0:, :, :] = 0.0
        data[:, :bw1, :] = 0.0
        data[:, -bw1:, :] = 0.0
        data[:, :, :bw2] = 0.0
        data[:, :, -bw2:] = 0.0
    else:
        raise ValueError(f"Only 1D, 2D, and 3D tensors supported, got {ndim}D")

    return data


def write_tensor_file(path, data):
    """Write a numpy float64 array as a binary tensor file (type_id=1 = double)."""
    data = np.asarray(data, dtype=np.float64)
    shape = data.shape
    ndim = len(shape)
    header = struct.pack(f"<{'Q' * (2 + ndim)}", ndim, 1, *shape)
    with open(path, "wb") as f:
        f.write(header)
        data.tofile(f)


def generate_stencil_spec(
    shape,
    negative_coeffs=True,
    original_precision="double",
    requested_slices=None,
    stencil_num=0,
    offsets=None,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(0)

    coeffs = None
    if requested_slices:
        abs_limit = 2 ** (7 * requested_slices)
        coeffs = rng.integers(
            -abs_limit if negative_coeffs else 0, abs_limit, size=shape
        ).astype(PRECISION_TYPE_MAP[original_precision])
        basis = rng.random(size=1, dtype=np.float32).astype(
            PRECISION_TYPE_MAP[original_precision]
        )
        coeffs *= basis
    else:
        coeffs = 1.0 - 2 * rng.random(
            size=shape, dtype=PRECISION_TYPE_MAP[original_precision]
        )
        coeffs = np.abs(coeffs) if not negative_coeffs else coeffs

    # divide by norm such that actual stencil is 5% within 1
    # scale_off = 0.05 * rng.random()
    scale_off = 0.0
    coeffs /= (1 + scale_off) * np.sum(np.abs(coeffs))

    if not offsets:
        offsets = np.array(shape) // 2
    shape = [str(x) for x in shape]
    offsets = [str(x) for x in offsets]
    coeffs_str = [str(x) for x in coeffs.flatten()]

    name = f"{'x'.join(shape)}{'Pos' if not negative_coeffs else ''}m{requested_slices if requested_slices else 'None'}_{stencil_num}"

    spec = f"""
    name {name}
    shape {" ".join(shape)}
    offsets {" ".join(offsets)}
    precision {original_precision}
    coeffs
    {" ".join(coeffs_str)}
    """
    with open(f"{STENCIL_DIR}/{name}.stc", "w") as outfile:
        outfile.write(spec)
    return name, coeffs


def run_pipeline(name, input_file, input_label, T):
    """
    Run compute-reference -> run-ozaki -> compute-errors for one (stencil, input, T).
    Returns (success: bool, error: float, oz_output: str) where oz_output is the
    combined stdout+stderr of run-ozaki, for diagnostics when error is large.
    """
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
            "--slice-limit",
            "12",
            "--extra-lsb",
            str(EXTRA_LSBS),
            "--extra-msb",
            str(EXTRA_MSBS),
            "--save-tensors",
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
            [f"{BUILD_BIN}/compute-errors", "-e", ref_file, "-a", oz_file],
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

    # print(", ".join([f"(S={i + 2} Err={e})" for i, e in enumerate(errs)]))
    return True, errs, oz_output


# --- Setup ---
os.makedirs(STENCIL_DIR, exist_ok=True)
os.makedirs(TENSOR_DIR, exist_ok=True)

# Generate one input tensor per unique (label, shape) in SHAPE_INPUT_PAIRS
input_tensors = {}  # label -> path
np.random.seed(0)
rng = np.random.default_rng(0)
seen_labels = {}  # label -> maximum stencil_shape for that label

# First pass: identify the maximum stencil shape for each label
for stencil_shape, input_shape, label in SHAPE_INPUT_PAIRS:
    if label not in seen_labels:
        seen_labels[label] = stencil_shape
    else:
        # Use the stencil shape with maximum extent (sum of sizes)
        if sum(stencil_shape) > sum(seen_labels[label]):
            seen_labels[label] = stencil_shape

# Second pass: generate tensors
for label, max_stencil_shape in seen_labels.items():
    # Find the input_shape for this label
    input_shape = None
    for stencil_shape, ishape, lbl in SHAPE_INPUT_PAIRS:
        if lbl == label:
            input_shape = ishape
            break

    if input_shape is None:
        continue

    path = os.path.join(DATA_DIR, f"{label}.dat")
    data = rng.standard_normal(input_shape)

    # Apply Dirichlet zero boundary condition if enabled
    if DIRICHLET_ZERO:
        data = apply_dirichlet_boundary(data, max_stencil_shape)
        print(
            f"Generated {label} tensor {input_shape} with Dirichlet boundary (stencil {max_stencil_shape}) -> {path}"
        )
    else:
        print(f"Generated {label} tensor {input_shape} -> {path}")

    write_tensor_file(path, data)
    input_tensors[label] = path

# --- Precision test loop ---
with (
    open(CSV_OUTPUT, "w", newline="") as csvfile,
    open(BREACHES_OUTPUT, "w", newline="") as breachfile,
):
    writer = csv.writer(csvfile)
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
            "rmse",
        ]
    )

    for stencil_shape, input_shape, input_label in SHAPE_INPUT_PAIRS:
        for neg_coeffs, req_slices, stencil_num in product(
            NEGATIVE_COEFFS, REQUESTED_SLICES, STENCIL_NAME
        ):
            name, coeffs = generate_stencil_spec(
                shape=stencil_shape,
                negative_coeffs=neg_coeffs,
                requested_slices=req_slices,
                stencil_num=stencil_num,
                rng=rng,
            )
            input_file = input_tensors[input_label]
            for T in TIMESTEPS:
                print(f"Testing {name} | {input_label} | T={T} ...")
                success, errs, oz_output = run_pipeline(
                    name, input_file, input_label, T
                )
                high_error = errs[-1] > ERROR_THRESHOLD
                stencil_msb_shift = oz_output.split("\n")[0][-1]
                writer.writerow(
                    [
                        name,
                        input_label,
                        T,
                        EXTRA_MSBS,
                        EXTRA_LSBS,
                        stencil_msb_shift,
                        "Completed" if success else "Failed",
                        "Wrong" if high_error else "Correct",
                        errs[-1],
                    ]
                )
                csvfile.flush()
                if success:
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
