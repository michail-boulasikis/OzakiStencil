"""
Diagnostic: compute RMSE separately for boundary vs interior elements.
Handles __float128 (IEEE 754 quad precision, 16 bytes) tensor format.
"""
import numpy as np
import struct
import os


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
        return (-1 if sign else 1) * 2**(-16382) * mantissa
    # Normal: value = (-1)^sign * 2^(exp-16383) * (1 + frac/2^112)
    mantissa = 1.0 + (frac_hi * (2**64) + frac_lo) / (2**112)
    value = (-1 if sign else 1) * (2 ** (exp - 16383)) * mantissa
    return value


def read_tensor(path):
    with open(path, "rb") as f:
        raw = f.read()
    ndim, type_id = struct.unpack("<QQ", raw[:16])
    shape = struct.unpack(f"<{'Q'*ndim}", raw[16: 16 + 8*ndim])
    offset = 16 + 8 * ndim
    n = 1
    for s in shape: n *= s

    if type_id == 0:   # float32
        data = np.frombuffer(raw[offset:], dtype=np.float32).copy().astype(np.float64)
    elif type_id == 1: # float64
        data = np.frombuffer(raw[offset:], dtype=np.float64).copy()
    elif type_id == 2: # long double (80-bit, stored as 16 bytes on Linux)
        data = np.frombuffer(raw[offset:], dtype=np.longdouble).copy().astype(np.float64)
    elif type_id == 3: # __float128 (16 bytes, IEEE quad precision)
        data = np.array([fp128_to_float64(raw[offset + 16*i: offset + 16*i + 16]) for i in range(n)])
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


def analyze(ref_path, oz_path, stencil_offsets, label):
    ref = read_tensor(ref_path)
    oz = read_tensor(oz_path)
    assert ref.shape == oz.shape, f"Shape mismatch: {ref.shape} vs {oz.shape}"

    interior_mask, boundary_mask = make_boundary_mask(ref.shape, stencil_offsets)
    err = ref - oz
    n_bnd = np.sum(boundary_mask)
    n_int = np.sum(interior_mask)
    n_tot = err.size

    rmse_total = rmse(err)
    rmse_bnd   = rmse(err[boundary_mask]) if n_bnd > 0 else 0.0
    rmse_int   = rmse(err[interior_mask]) if n_int > 0 else 0.0
    bnd_frac   = n_bnd / n_tot

    print(f"{label:<30} total={rmse_total:.4e}  bnd={rmse_bnd:.4e}  int={rmse_int:.4e}  "
          f"bnd_frac={100*bnd_frac:.1f}%  n_bnd={n_bnd}")


TENSOR_DIR = "./data/tensors"

cases = [
    # (ref_file, oz_file_suffix, stencil_offsets, label)
    # 1D stencil size 5 -> offset 2
    ("ref_1D_128_5m1_0_fp128_1.dat",      "oz_1D_128_5m1_0_1_8.dat",      [2],      "1D 5m1_neg S=8"),
    ("ref_1D_128_5Posm1_0_fp128_1.dat",   "oz_1D_128_5Posm1_0_1_8.dat",   [2],      "1D 5m1_pos S=8"),
    ("ref_1D_128_5mNone_0_fp128_1.dat",   "oz_1D_128_5mNone_0_1_8.dat",   [2],      "1D 5mNone_neg S=8"),
    ("ref_1D_128_5PosmNone_0_fp128_1.dat","oz_1D_128_5PosmNone_0_1_8.dat",[2],      "1D 5mNone_pos S=8"),
    # 2D stencil 3x3 -> offset [1,1]
    ("ref_2D_128x128_3x3m1_0_fp128_1.dat",     "oz_2D_128x128_3x3m1_0_1_9.dat",     [1,1], "2D 3x3m1_neg S=9"),
    ("ref_2D_128x128_3x3Posm1_0_fp128_1.dat",  "oz_2D_128x128_3x3Posm1_0_1_9.dat",  [1,1], "2D 3x3m1_pos S=9"),
    ("ref_2D_128x128_3x3mNone_0_fp128_1.dat",  "oz_2D_128x128_3x3mNone_0_1_9.dat",  [1,1], "2D 3x3mNone_neg S=9"),
    ("ref_2D_128x128_3x3PosmNone_0_fp128_1.dat","oz_2D_128x128_3x3PosmNone_0_1_9.dat",[1,1],"2D 3x3mNone_pos S=9"),
    # 3D stencil 3x3x3 -> offset [1,1,1]
    ("ref_3D_32x32x32_3x3x3m1_0_fp128_1.dat",      "oz_3D_32x32x32_3x3x3m1_0_1_10.dat",      [1,1,1], "3D 3^3m1_neg S=10"),
    ("ref_3D_32x32x32_3x3x3Posm1_0_fp128_1.dat",   "oz_3D_32x32x32_3x3x3Posm1_0_1_10.dat",   [1,1,1], "3D 3^3m1_pos S=10"),
    ("ref_3D_32x32x32_3x3x3mNone_0_fp128_1.dat",   "oz_3D_32x32x32_3x3x3mNone_0_1_10.dat",   [1,1,1], "3D 3^3mNone_neg S=10"),
    ("ref_3D_32x32x32_3x3x3PosmNone_0_fp128_1.dat","oz_3D_32x32x32_3x3x3PosmNone_0_1_10.dat",[1,1,1], "3D 3^3mNone_pos S=10"),
]

print(f"{'Label':<30} {'total':>10}  {'bnd':>10}  {'int':>10}  bnd_frac  n_bnd")
print("-" * 90)
for ref_name, oz_name, offsets, label in cases:
    ref_path = os.path.join(TENSOR_DIR, ref_name)
    oz_path  = os.path.join(TENSOR_DIR, oz_name)
    if not os.path.exists(ref_path):
        print(f"{label:<30} REF MISSING")
        continue
    if not os.path.exists(oz_path):
        print(f"{label:<30} OZ MISSING: {oz_name}")
        continue
    try:
        analyze(ref_path, oz_path, offsets, label)
    except Exception as e:
        print(f"{label:<30} ERROR: {e}")
