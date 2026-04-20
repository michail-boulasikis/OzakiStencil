#!/usr/bin/env bash
# correctness.sh - End-to-end correctness pipeline for ozaki-stencil
#
# Usage: ./correctness.sh <stencil1.stc> [stencil2.stc ...]
#
# Pipeline:
#   1. Build check - aborts if any required binary is missing
#   2. Input tensor generation - one FP128 tensor per dimensionality (1D/2D/3D)
#   3. CPU reference computation - FP32, FP64, FP80, FP128 for all timesteps
#   4. GPU Ozaki solver - slices 2..MAX_SLICES, all timesteps, save tensors
#   5. RMSE computation - Ozaki vs FP128 (slices 5..MAX_SLICES) and
#                         FP32/FP64/FP80 vs FP128 (for the refs plot)
#   6. RMSE plot via scripts/plot_rmse.py

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
readonly TIMESTEPS=(1 10 100 1000)
readonly MAX_SLICES=18
readonly MIN_SLICES=5
readonly BUILD_BIN="./build/bin"
readonly DATA_DIR="./data"
readonly TENSOR_DIR="./data/tensors"
readonly RESULTS_DIR="./data/results"
readonly MAIN_CSV="${RESULTS_DIR}/correctness_results.csv"
readonly REFS_CSV="${RESULTS_DIR}/precision_refs.csv"
readonly PLOT_OUT="${RESULTS_DIR}/correctness_rmse.eps"

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $0 <stencil1.stc> [stencil2.stc ...]

  stencil.stc  One or more stencil specification files

Example:
  $0 data/stencils/7x7Lm1_0.stc data/stencils/7x7NmNone_0.stc
EOF
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

STENCIL_FILES=("$@")

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
log()      { echo "[$(date '+%H:%M:%S')] $*"; }
log_warn() { echo "[$(date '+%H:%M:%S')] WARN: $*" >&2; }
log_err()  { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; }

# ---------------------------------------------------------------------------
# Step 1: Build check
# ---------------------------------------------------------------------------
log "Checking for compiled binaries..."
NEED_BUILD=0
for bin in compute-reference run-ozaki compute-errors gen-tensor; do
    if [[ ! -x "${BUILD_BIN}/${bin}" ]]; then
        log "  Missing: ${BUILD_BIN}/${bin}"
        NEED_BUILD=1
    fi
done

if [[ "$NEED_BUILD" -eq 1 ]]; then
    log_err "One or more required binaries are missing. Rebuild with: ./build.sh -t -- -DOZAKI_USE_BOOST_MULTIPRECISION=ON"
    exit 1
else
    log "All binaries present."
fi

# ---------------------------------------------------------------------------
# Helpers: stencil file parsing
# ---------------------------------------------------------------------------

# Extract the stencil name from the `name` line of a .stc file
parse_stencil_name() {
    grep -m1 '^\s*name\b' "$1" | awk '{print $2}'
}

# Count tokens after `shape` to determine dimensionality (1, 2, or 3)
parse_stencil_ndim() {
    grep -m1 '^\s*shape\b' "$1" | awk '{print NF - 1}'
}

# ---------------------------------------------------------------------------
# Helpers: tensor sizing
# ---------------------------------------------------------------------------

get_tensor_path() {
    case "$1" in
        1) echo "${DATA_DIR}/1D_262144.dat" ;;
        2) echo "${DATA_DIR}/2D_512x512.dat" ;;
        3) echo "${DATA_DIR}/3D_64x64x64.dat" ;;
        *) log_err "Unsupported dimensionality: $1"; exit 1 ;;
    esac
}

# Generate an input tensor for the given dimensionality using gen-tensor.
# Usage: generate_tensor <ndim> <output_path>
generate_tensor() {
    local ndim="$1" path="$2"
    case "$ndim" in
        1)
            "${BUILD_BIN}/gen-tensor" -o "${path}" --N 262144 --dims 1
            ;;
        2)
            "${BUILD_BIN}/gen-tensor" -o "${path}" --N 512 --dims 2
            ;;
        3)
            "${BUILD_BIN}/gen-tensor" -o "${path}" --N 64  --dims 3
            ;;
        *)
            log_err "Unsupported dimensionality for tensor generation: $ndim"
            exit 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Helper: safe RMSE computation
# Returns "nan" when a file is missing or compute-errors fails
# ---------------------------------------------------------------------------
compute_rmse() {
    local ref="$1" actual="$2"
    if [[ ! -f "$ref" || ! -f "$actual" ]]; then
        echo "nan"
        return
    fi
    local result
    if result=$("${BUILD_BIN}/compute-errors" -e "$ref" -a "$actual" --scaled 2>/dev/null); then
        # strip trailing whitespace/newlines
        echo "${result}" | tr -d '[:space:]'
    else
        echo "nan"
    fi
}

# ---------------------------------------------------------------------------
# Step 2: Validate stencil files and generate input tensors
# ---------------------------------------------------------------------------
log ""
log "Validating stencil files and generating input tensors..."
mkdir -p "${TENSOR_DIR}"
mkdir -p "${RESULTS_DIR}"

declare -A TENSOR_DONE   # ndim -> 1 once tensor has been generated/verified

for stencil_file in "${STENCIL_FILES[@]}"; do
    if [[ ! -f "$stencil_file" ]]; then
        log_err "Stencil file not found: $stencil_file"
        exit 1
    fi

    ndim=$(parse_stencil_ndim "$stencil_file")
    if [[ -z "$ndim" || "$ndim" -lt 1 || "$ndim" -gt 3 ]]; then
        log_err "Could not parse dimensionality from: $stencil_file"
        exit 1
    fi

    tensor_path=$(get_tensor_path "$ndim")

    if [[ -z "${TENSOR_DONE[$ndim]+x}" ]]; then
        if [[ -f "$tensor_path" ]]; then
            log "  Reusing existing tensor: $tensor_path"
        else
            log "  Generating ${ndim}D tensor -> $tensor_path"
            generate_tensor "$ndim" "$tensor_path"
        fi
        TENSOR_DONE[$ndim]=1
    fi
done

# ---------------------------------------------------------------------------
# Step 3–5: Per-stencil reference computation, Ozaki runs, error computation
# ---------------------------------------------------------------------------
log ""
log "Initialising output CSVs..."
echo "stencil,T,slices,rmse"     > "${MAIN_CSV}"
echo "stencil,T,precision,rmse"  > "${REFS_CSV}"

for stencil_file in "${STENCIL_FILES[@]}"; do
    name=$(parse_stencil_name "$stencil_file")
    ndim=$(parse_stencil_ndim "$stencil_file")
    tensor_path=$(get_tensor_path "$ndim")

    log ""
    log "================================================================"
    log "Stencil: ${name}  (${ndim}D)  file: ${stencil_file}"
    log "================================================================"

    ALGO="direct"
    if [[ ${name} == 7x7* ]] ; then
      ALGO="convstencil"
    fi

    # ------------------------------------------------------------------
    # 3a: CPU reference — all four precisions, all timesteps, one call
    # ------------------------------------------------------------------
    log "  [ref] Computing CPU references (fp32 / fp64 / fp80 / fp128)..."
    "${BUILD_BIN}/compute-reference" \
        -i  "${tensor_path}" \
        --stencil-file "${stencil_file}" \
        -T  "${TIMESTEPS[@]}" \
        --float --double --longdouble --fp128

    # ------------------------------------------------------------------
    # 3b: GPU Ozaki solver — all timesteps, slices 2..MAX_SLICES, one call
    # ------------------------------------------------------------------
    log "  [ozaki] Running GPU Ozaki solver (slices 2..${MAX_SLICES})..."
    "${BUILD_BIN}/run-ozaki" \
        -i  "${tensor_path}" \
        --stencil-file "${stencil_file}" \
        -T  "${TIMESTEPS[@]}" \
        -A "${ALGO}" \
        --slice-limit "${MAX_SLICES}" \
        --save-tensors

    # ------------------------------------------------------------------
    # 3c: Error computation
    # ------------------------------------------------------------------
    log "  [errors] Computing RMSE..."

    for T in "${TIMESTEPS[@]}"; do
        ref_fp128="${TENSOR_DIR}/ref_${name}_fp128_${T}.dat"

        if [[ ! -f "${ref_fp128}" ]]; then
            log_warn "Missing fp128 reference for T=${T}: ${ref_fp128} — skipping."
            continue
        fi

        # Ozaki vs fp128: slices MIN_SLICES..MAX_SLICES
        for s in $(seq "${MIN_SLICES}" "${MAX_SLICES}"); do
            oz_file="${TENSOR_DIR}/oz_${name}_${T}_${s}.dat"
            rmse=$(compute_rmse "${ref_fp128}" "${oz_file}")
            if [[ "$rmse" != "nan" ]]; then
                printf '%s,%s,%s,%s\n' "${name}" "${T}" "${s}" "${rmse}" >> "${MAIN_CSV}"
            fi
        done

        # Precision references vs fp128: float / double / longdouble
        for prec in float double longdouble; do
            ref_prec="${TENSOR_DIR}/ref_${name}_${prec}_${T}.dat"
            rmse=$(compute_rmse "${ref_fp128}" "${ref_prec}")
            if [[ "$rmse" != "nan" ]]; then
                printf '%s,%s,%s,%s\n' "${name}" "${T}" "${prec}" "${rmse}" >> "${REFS_CSV}"
            fi
        done

        log "    T=${T}: done"
    done

    log "  Stencil ${name} complete."
done

# ---------------------------------------------------------------------------
# Step 4: Plot
# ---------------------------------------------------------------------------
log ""
log "Generating RMSE plot -> ${PLOT_OUT}"
python3 scripts/plot_rmse.py "${MAIN_CSV}" \
    --refs   "${REFS_CSV}" \
    --output "${PLOT_OUT}"

log ""
log "================================================================"
log "Correctness pipeline complete."
log "  Main CSV : ${MAIN_CSV}"
log "  Refs CSV : ${REFS_CSV}"
log "  Plot     : ${PLOT_OUT}"
log "================================================================"
