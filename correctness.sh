#!/usr/bin/env bash
# correctness.sh - End-to-end correctness pipeline for ozaki-stencil
#
# Usage: ./correctness.sh [--skip-refs] [--longer-loops] [stencil1.stc ...]
#
# Pipeline:
#   1. Build check - aborts if any required binary is missing
#   2. Input tensor generation for each supported dimensionality (2D/3D)
#   3. CPU reference computation at the precisions supported by the target SM
#   4. GPU Ozaki solver - slices 2..MAX_SLICES, all timesteps, save tensors
#   5. Error computation against FP64 below SM100 and FP128 on SM100+
#   6. Error plots via scripts/plot_rmse.py

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
readonly BASE_TIMESTEPS=(1 10 100)
readonly MIN_SLICES=5
readonly BUILD_BIN="./build/bin"
readonly DATA_DIR="./data"
readonly TENSOR_DIR="./data/tensors"
readonly RESULTS_DIR="./data/results"
readonly MAIN_CSV="${RESULTS_DIR}/correctness_results.csv"
readonly REFS_CSV="${RESULTS_DIR}/precision_refs.csv"
readonly PLOT_OUT="${RESULTS_DIR}/correctness_rmse.eps"

if [[ -z "${CUDA_COMPUTE_CAPABILITY:-}" ]]; then
    echo "ERROR: CUDA_COMPUTE_CAPABILITY must be set (for example, 120)." >&2
    exit 1
fi
if [[ ! "${CUDA_COMPUTE_CAPABILITY}" =~ ^([0-9]+)[[:alpha:]]*$ ]]; then
    echo "ERROR: CUDA_COMPUTE_CAPABILITY must contain a numeric SM followed by an optional architecture suffix; got '${CUDA_COMPUTE_CAPABILITY}'." >&2
    exit 1
fi

readonly COMPUTE_CAPABILITY=$((10#${BASH_REMATCH[1]}))
if (( COMPUTE_CAPABILITY < 100 )); then
    readonly MAX_SLICES=12
    readonly REFERENCE_PRECISION="double"
    readonly -a REFERENCE_OPTIONS=(--float --double --longdouble)
    readonly -a COMPARISON_PRECISIONS=(float longdouble)
else
    readonly MAX_SLICES=18
    readonly REFERENCE_PRECISION="fp128"
    readonly -a REFERENCE_OPTIONS=(--float --double --longdouble --fp128)
    readonly -a COMPARISON_PRECISIONS=(float double longdouble)
fi

# Three stencils from every supported shape: one-, two-, and nine-slice
# (mNone) coefficients. The remaining tags deliberately mix coefficient sign,
# sparsity, and L1-norm categories. Positional stencil arguments replace this
# default set.
readonly -a DEFAULT_STENCIL_FILES=(
    "${DATA_DIR}/stencils/3x3SpNm1_0.stc"
    "${DATA_DIR}/stencils/3x3Lm2_0.stc"
    "${DATA_DIR}/stencils/3x3PosLmNone_0.stc"
    "${DATA_DIR}/stencils/7x7Lm1_0.stc"
    "${DATA_DIR}/stencils/7x7SpLm2_0.stc"
    "${DATA_DIR}/stencils/7x7PosSpNmNone_0.stc"
    "${DATA_DIR}/stencils/15x15Lm1_0.stc"
    "${DATA_DIR}/stencils/15x15PosSpNm2_0.stc"
    "${DATA_DIR}/stencils/15x15LmNone_0.stc"
    "${DATA_DIR}/stencils/3x3x3Lm1_0.stc"
    "${DATA_DIR}/stencils/3x3x3PosNm2_0.stc"
    "${DATA_DIR}/stencils/3x3x3PosLmNone_0.stc"
    "${DATA_DIR}/stencils/5x5x5Lm1_0.stc"
    "${DATA_DIR}/stencils/5x5x5PosLm2_0.stc"
    "${DATA_DIR}/stencils/5x5x5SpLmNone_0.stc"
)

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $0 [--skip-refs] [--longer-loops] [stencil1.stc ...]

  --skip-refs   Skip CPU reference construction and reuse existing references
  --longer-loops  Also run the 10000 and 100000 timestep cases
  stencil.stc  One or more stencil specification files. If omitted, run three
               representative stencils from every supported shape.

Example:
  $0
  $0 data/stencils/7x7Lm1_0.stc data/stencils/7x7NmNone_0.stc
EOF
    exit "${1:-1}"
}

SKIP_REFS=0
LONGER_LOOPS=0
STENCIL_FILES=()
for arg in "$@"; do
    case "$arg" in
        --skip-refs)
            SKIP_REFS=1
            ;;
        --longer-loops)
            LONGER_LOOPS=1
            ;;
        -h|--help)
            usage 0
            ;;
        *)
            STENCIL_FILES+=("$arg")
            ;;
    esac
done

USING_DEFAULT_STENCILS=0
if [[ "${#STENCIL_FILES[@]}" -eq 0 ]]; then
    STENCIL_FILES=("${DEFAULT_STENCIL_FILES[@]}")
    USING_DEFAULT_STENCILS=1
fi

TIMESTEPS=("${BASE_TIMESTEPS[@]}")
if [[ "$LONGER_LOOPS" -eq 1 ]]; then
    TIMESTEPS+=(1000 10000 100000)
fi

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
REQUIRED_BINS=(run-ozaki compute-errors gen-tensor)
if [[ "$SKIP_REFS" -eq 0 ]]; then
    REQUIRED_BINS+=(compute-reference)
fi
for bin in "${REQUIRED_BINS[@]}"; do
    if [[ ! -x "${BUILD_BIN}/${bin}" ]]; then
        log "  Missing: ${BUILD_BIN}/${bin}"
        NEED_BUILD=1
    fi
done

if [[ "$NEED_BUILD" -eq 1 ]]; then
    log_err "One or more required binaries are missing. Rebuild with: ./build.sh -t"
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

# The halo (zero boundary) of a stencil: half its side.
parse_stencil_halo() {
    grep -m1 '^\s*shape\b' "$1" | awk '{print int($2 / 2)}'
}

# Input tensor for a stencil: Gaussian values whose zero boundary is exactly
# the stencil's halo, so no interior cell is artificially zero. All variants
# share the seed and hence their interior values.
# Usage: get_tensor_path <ndim> <halo>
get_tensor_path() {
    case "$1" in
        2) echo "${DATA_DIR}/2D_512x512_halo$2.dat" ;;
        3) echo "${DATA_DIR}/3D_64x64x64_halo$2.dat" ;;
        *) log_err "Unsupported dimensionality: $1"; exit 1 ;;
    esac
}

# Usage: generate_tensor <ndim> <halo> <output_path>
generate_tensor() {
    local ndim="$1" halo="$2" path="$3"
    case "$ndim" in
        2) "${BUILD_BIN}/gen-tensor" -o "${path}" --N 512 --dims 2 --halo "${halo}" ;;
        3) "${BUILD_BIN}/gen-tensor" -o "${path}" --N 64  --dims 3 --halo "${halo}" ;;
        *) log_err "Unsupported dimensionality for tensor generation: $ndim"; exit 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Helper: safe error computation
# Returns "nan,nan" when a file is missing or compute-errors fails
# ---------------------------------------------------------------------------
compute_errors() {
    local ref="$1" actual="$2"
    if [[ ! -f "$ref" || ! -f "$actual" ]]; then
        echo "nan,nan"
        return
    fi
    local result
    if result=$("${BUILD_BIN}/compute-errors" -e "$ref" -a "$actual" --scaled 2>/dev/null); then
        # strip trailing whitespace/newlines
        result=$(echo "${result}" | tr -d '[:space:]')
        if [[ "$result" == *,* ]]; then
            echo "${result}"
        else
            echo "nan,nan"
        fi
    else
        echo "nan,nan"
    fi
}

# ---------------------------------------------------------------------------
# Helper: final-slice error summary
# ---------------------------------------------------------------------------
# The CSV retains every slice in the sweep. This compact terminal table reports
# the final (and highest-precision) configured slice count for each stencil and
# timestep; unavailable measurements are shown as n/a.
print_error_summary() {
    local stencil_file name T values rmse max_abs_diff
    local stencil_width=28
    local metric_width=15
    local timestep_width=$((2 * metric_width + 1))

    printf '\nCollected Ozaki errors at %s slices\n' "${MAX_SLICES}"
    printf '%-*s' "${stencil_width}" "Stencil"
    for T in "${TIMESTEPS[@]}"; do
        printf '%-*s' "${timestep_width}" "T=${T}"
    done
    printf '\n'

    printf '%-*s' "${stencil_width}" ""
    for T in "${TIMESTEPS[@]}"; do
        printf '%-*s %-*s' "${metric_width}" "RMSE" "${metric_width}" "Max abs diff"
    done
    printf '\n'

    for stencil_file in "${STENCIL_FILES[@]}"; do
        name=$(parse_stencil_name "${stencil_file}")
        printf '%-*s' "${stencil_width}" "${name}"
        for T in "${TIMESTEPS[@]}"; do
            values=$(awk -F, -v stencil="${name}" -v timestep="${T}" -v slices="${MAX_SLICES}" '
                NR > 1 && $1 == stencil && $2 == timestep && $3 == slices {
                    rmse = $4
                    max_abs_diff = $5
                }
                END {
                    if (rmse != "" && max_abs_diff != "")
                        printf "%s,%s", rmse, max_abs_diff
                    else
                        printf "n/a,n/a"
                }
            ' "${MAIN_CSV}")
            IFS=',' read -r rmse max_abs_diff <<< "${values}"
            printf '%-*s %-*s' "${metric_width}" "${rmse}" "${metric_width}" "${max_abs_diff}"
        done
        printf '\n'
    done
}

# ---------------------------------------------------------------------------
# Step 2: Validate stencil files and generate input tensors
# ---------------------------------------------------------------------------
log ""
log "Validating stencil files and generating input tensors..."
if [[ "$USING_DEFAULT_STENCILS" -eq 1 ]]; then
    log "Using the default ${#STENCIL_FILES[@]}-stencil correctness suite."
fi
mkdir -p "${TENSOR_DIR}"
mkdir -p "${RESULTS_DIR}"

declare -A TENSOR_DONE   # tensor path -> 1 once generated/verified

for stencil_file in "${STENCIL_FILES[@]}"; do
    if [[ ! -f "$stencil_file" ]]; then
        log_err "Stencil file not found: $stencil_file"
        exit 1
    fi

    ndim=$(parse_stencil_ndim "$stencil_file")
    if [[ -z "$ndim" || "$ndim" -lt 2 || "$ndim" -gt 3 ]]; then
        log_err "Could not parse dimensionality from: $stencil_file"
        exit 1
    fi

    halo=$(parse_stencil_halo "$stencil_file")
    tensor_path=$(get_tensor_path "$ndim" "$halo")

    if [[ -z "${TENSOR_DONE[$tensor_path]+x}" ]]; then
        if [[ -f "$tensor_path" ]]; then
            log "  Reusing existing tensor: $tensor_path"
        else
            log "  Generating ${ndim}D tensor (halo ${halo}) -> $tensor_path"
            generate_tensor "$ndim" "$halo" "$tensor_path"
        fi
        TENSOR_DONE[$tensor_path]=1
    fi
done

# ---------------------------------------------------------------------------
# Step 3–5: Per-stencil reference computation, Ozaki runs, error computation
# ---------------------------------------------------------------------------
log ""
log "Initialising output CSVs..."
echo "stencil,T,slices,rmse,max_abs_diff"    > "${MAIN_CSV}"
echo "stencil,T,precision,rmse,max_abs_diff" > "${REFS_CSV}"

for stencil_file in "${STENCIL_FILES[@]}"; do
    name=$(parse_stencil_name "$stencil_file")
    ndim=$(parse_stencil_ndim "$stencil_file")
    tensor_path=$(get_tensor_path "$ndim" "$(parse_stencil_halo "$stencil_file")")
    # Output names carry the input tensor, so results computed from another
    # input (e.g. a different zero-boundary width) are never reused.
    input_tag=$(basename "${tensor_path}" .dat)

    log ""
    log "================================================================"
    log "Stencil: ${name}  (${ndim}D)  file: ${stencil_file}"
    log "================================================================"

    # ------------------------------------------------------------------
    # 3a: CPU references supported by this architecture, all timesteps
    # ------------------------------------------------------------------
    if [[ "$SKIP_REFS" -eq 0 ]]; then
        log "  [ref] Computing CPU references (comparison target: ${REFERENCE_PRECISION})..."
        "${BUILD_BIN}/compute-reference" \
            -i  "${tensor_path}" \
            --stencil-file "${stencil_file}" \
            --prefix "ref_${input_tag}" \
            -T  "${TIMESTEPS[@]}" \
            "${REFERENCE_OPTIONS[@]}"
    else
        log "  [ref] Skipping CPU reference construction (--skip-refs)."
    fi

    # ------------------------------------------------------------------
    # 3b: GPU Ozaki solver — all timesteps, slices 2..MAX_SLICES, one call
    # ------------------------------------------------------------------
    log "  [ozaki] Running GPU Ozaki solver (slices 2..${MAX_SLICES})..."
    "${BUILD_BIN}/run-ozaki" \
        -i  "${tensor_path}" \
        --stencil-file "${stencil_file}" \
        --prefix "oz_${input_tag}" \
        -T  "${TIMESTEPS[@]}" \
        --slice-limit "${MAX_SLICES}"

    # ------------------------------------------------------------------
    # 3c: Error computation
    # ------------------------------------------------------------------
    log "  [errors] Computing RMSE and maximum absolute difference..."

    for T in "${TIMESTEPS[@]}"; do
        reference_file="${TENSOR_DIR}/ref_${input_tag}_${name}_${REFERENCE_PRECISION}_${T}.dat"

        if [[ ! -f "${reference_file}" ]]; then
            log_warn "Missing ${REFERENCE_PRECISION} reference for T=${T}: ${reference_file} — skipping."
            continue
        fi

        # Ozaki vs the architecture-appropriate reference.
        for s in $(seq "${MIN_SLICES}" "${MAX_SLICES}"); do
            oz_file="${TENSOR_DIR}/oz_${input_tag}_${name}_${T}_${s}.dat"
            IFS=',' read -r rmse max_abs_diff < <(compute_errors "${reference_file}" "${oz_file}")
            if [[ "$rmse" != "nan" && "$max_abs_diff" != "nan" ]]; then
                printf '%s,%s,%s,%s,%s\n' \
                    "${name}" "${T}" "${s}" "${rmse}" "${max_abs_diff}" >> "${MAIN_CSV}"
            fi
        done

        # Lower/alternate precision baselines vs the selected reference.
        for prec in "${COMPARISON_PRECISIONS[@]}"; do
            ref_prec="${TENSOR_DIR}/ref_${input_tag}_${name}_${prec}_${T}.dat"
            IFS=',' read -r rmse max_abs_diff < <(compute_errors "${reference_file}" "${ref_prec}")
            if [[ "$rmse" != "nan" && "$max_abs_diff" != "nan" ]]; then
                printf '%s,%s,%s,%s,%s\n' \
                    "${name}" "${T}" "${prec}" "${rmse}" "${max_abs_diff}" >> "${REFS_CSV}"
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
log "Generating error plots -> ${PLOT_OUT}"
python3 scripts/plot_rmse.py "${MAIN_CSV}" \
    --refs   "${REFS_CSV}" \
    --output "${PLOT_OUT}"

print_error_summary

log ""
log "================================================================"
log "Correctness pipeline complete."
log "  Main CSV : ${MAIN_CSV}"
log "  Refs CSV : ${REFS_CSV}"
log "  Plot     : ${PLOT_OUT}"
log "================================================================"
