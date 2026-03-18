#!/usr/bin/env bash
# ozaki_worth_table.sh — Benchmark table: OzakiStencil Direct vs FP32/FP64/FP128 baselines
#
# Sweeps Ozaki Direct over data slices s=SLICE_MIN..SLICE_MAX and M=1,2,3,None,
# pairing each M value with its corresponding stencil file (<base>m{M}_0.stc).
# Outputs a TSV with a comment header containing the naive baseline GStPt/s values.
#
# Usage:
#   ./scripts/ozaki_worth_table.sh <stencil_base> <domain_size> [options]
#
#   stencil_base   Name prefix, e.g. "5x5L" maps to 5x5Lm1_0.stc, 5x5Lm2_0.stc, ...
#   domain_size    Grid shape, e.g. 4096x4096
#
# Options:
#   --stencil-dir DIR   Stencil file directory        (default: <repo>/data/stencils)
#   --bench BIN         bench-stencil binary path     (default: <repo>/build/bin/bench-stencil)
#   --slice-min N       Minimum data slice count      (default: 2)
#   --slice-max N       Maximum data slice count      (default: 18)
#   -T N                Integration timesteps         (default: 20)
#   -W N                Warmup timesteps              (default: 1)
#   -R N                Measurement repeats           (default: 3)
#   -o FILE             Output TSV file               (default: <base>_<domain>_table.tsv in scripts/)
#
# Output TSV columns: slices, M=1, M=2, M=3, M=None  (GStPt/s)
# Comment lines (#) at the top carry FP32/FP64/FP128 baselines and run metadata.
#
# Quick postprocess in Python:
#   import pandas as pd
#   df = pd.read_csv('5x5L_4096x4096_table.tsv', sep='\t', comment='#')

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------- defaults ----------
BENCH="$REPO_DIR/build/bin/bench-stencil"
STENCIL_DIR="$REPO_DIR/data/stencils"
SLICE_MIN=2
SLICE_MAX=18
TIMESTEPS=20
WARMUP=1
REPEATS=3
OUTPUT_FILE=""

# ---------- positional args ----------
if [[ $# -lt 2 ]]; then
    grep '^#' "$0" | sed 's/^# \?//' | head -40
    exit 1
fi
STENCIL_BASE="$1"
DOMAIN="$2"
shift 2

# Select algorithm based on stencil dimensions
if [[ "$STENCIL_BASE" =~ ^7x7 ]]; then
    OZ_ALGO="convstencil"
    OZ_LABEL="Ozaki ConvStencil"
else
    OZ_ALGO="direct"
    OZ_LABEL="Ozaki Direct"
fi

# ---------- named options ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stencil-dir) STENCIL_DIR="$2"; shift 2 ;;
        --bench)       BENCH="$2";        shift 2 ;;
        --slice-min)   SLICE_MIN="$2";    shift 2 ;;
        --slice-max)   SLICE_MAX="$2";    shift 2 ;;
        -T)            TIMESTEPS="$2";    shift 2 ;;
        -W)            WARMUP="$2";       shift 2 ;;
        -R)            REPEATS="$2";      shift 2 ;;
        -o)            OUTPUT_FILE="$2";  shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$OUTPUT_FILE" ]] && OUTPUT_FILE="$SCRIPT_DIR/${STENCIL_BASE}_${DOMAIN}_table.tsv"

# ---------- stencil file map ----------
M_VALUES=(1 2 3 None)
declare -A STC_FILE
for m in "${M_VALUES[@]}"; do
    STC_FILE[$m]="$STENCIL_DIR/${STENCIL_BASE}m${m}_0.stc"
done

# Determine which M values actually have files
PRESENT_M=()
for m in "${M_VALUES[@]}"; do
    if [[ -f "${STC_FILE[$m]}" ]]; then
        PRESENT_M+=("$m")
    else
        echo "Warning: stencil file not found, skipping M=${m}: ${STC_FILE[$m]}" >&2
    fi
done

if [[ ${#PRESENT_M[@]} -eq 0 ]]; then
    echo "Error: no stencil files found for base '${STENCIL_BASE}' in '${STENCIL_DIR}'" >&2
    exit 1
fi

# ---------- helpers ----------

# Run bench-stencil and return its stdout; errors go to stderr
run_bench() {
    local stencil_file="$1"
    local algorithms="$2"
    local slices="${3:-0}"
    "$BENCH" \
        -S "$DOMAIN" \
        -A "$algorithms" \
        --stencil-file "$stencil_file" \
        -s "$slices" \
        -T "$TIMESTEPS" \
        -W "$WARMUP" \
        -R "$REPEATS"
}

# Extract GStPt/s (last field) from the line matching <keyword>
extract_gstp() {
    local output="$1"
    local keyword="$2"
    local val
    # keyword may contain spaces (e.g. "Naive FP32", "Ozaki Direct")
    val=$(printf '%s\n' "$output" | grep -E "^[[:space:]]*${keyword}[[:space:]]" | awk '{print $NF}' | head -1)
    printf '%s' "${val:-N/A}"
}

# ---------- step 1: naive baselines ----------
# M doesn't affect naive kernels; use the first available stencil file.
BASELINE_STC="${STC_FILE[${PRESENT_M[0]}]}"

TOTAL_RUNS=$(( (SLICE_MAX - SLICE_MIN + 1) * ${#PRESENT_M[@]} ))
echo "==> Naive baselines (FP32 / FP64 / FP128) using: $(basename "$BASELINE_STC")" >&2

NAIVE_OUT=$(run_bench "$BASELINE_STC" "naive-fp32,naive-fp64,naive-fp128" 0)

FP32_GSTP=$(extract_gstp  "$NAIVE_OUT" "Naive FP32")
FP64_GSTP=$(extract_gstp  "$NAIVE_OUT" "Naive FP64")
FP128_GSTP=$(extract_gstp "$NAIVE_OUT" "Naive FP128")

printf "    FP32:  %s GStPt/s\n"  "$FP32_GSTP"  >&2
printf "    FP64:  %s GStPt/s\n"  "$FP64_GSTP"  >&2
printf "    FP128: %s GStPt/s\n"  "$FP128_GSTP" >&2
echo >&2

# ---------- step 2: Ozaki Direct sweep ----------
echo "==> ${OZ_LABEL} sweep: s=${SLICE_MIN}..${SLICE_MAX}, M=(${PRESENT_M[*]})" >&2
echo "    ${TOTAL_RUNS} benchmark runs total" >&2
echo >&2

run_idx=0

{
    # --- TSV header (comment block) ---
    printf '# ozaki_worth_table stencil_base=%s domain=%s\n' "$STENCIL_BASE" "$DOMAIN"
    printf '# timesteps=%d warmup=%d repeats=%d slice_min=%d slice_max=%d\n' \
        "$TIMESTEPS" "$WARMUP" "$REPEATS" "$SLICE_MIN" "$SLICE_MAX"
    printf '# baseline_fp32=%s baseline_fp64=%s baseline_fp128=%s  (GStPt/s)\n' \
        "$FP32_GSTP" "$FP64_GSTP" "$FP128_GSTP"

    # --- column header ---
    printf 'slices'
    for m in "${PRESENT_M[@]}"; do
        printf '\tM=%s' "$m"
    done
    printf '\n'

    # --- data rows ---
    for s in $(seq "$SLICE_MIN" "$SLICE_MAX"); do
        row="$s"
        for m in "${PRESENT_M[@]}"; do
            run_idx=$(( run_idx + 1 ))
            printf '  [%d/%d] s=%d M=%s ...' "$run_idx" "$TOTAL_RUNS" "$s" "$m" >&2
            out=$(run_bench "${STC_FILE[$m]}" "$OZ_ALGO" "$s")
            gstp=$(extract_gstp "$out" "$OZ_LABEL")
            printf ' %s GStPt/s\n' "$gstp" >&2
            row="${row}	${gstp}"
        done
        printf '%s\n' "$row"
    done

} | tee "$OUTPUT_FILE"

echo >&2
echo "==> Done. Table written to: $OUTPUT_FILE" >&2

