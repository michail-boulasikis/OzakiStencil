#!/bin/bash
set -u # Error on undefined variables

# ==========================================
# --- 1. Configuration & Constants ---
# ==========================================

# Default Arguments
WARMUP_RUNS=10
ARCH="${CUDA_COMPUTE_CAPABILITY:?Error: CUDA_COMPUTE_CAPABILITY is not set. Run via entrypoint.sh or set it manually (e.g. export CUDA_COMPUTE_CAPABILITY=120).}"
CLEAN_FIRST=0
JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)

# Paths
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${ROOT_DIR}/benchmark"
REPORT_DIR="${ROOT_DIR}/data/results"
PATCH_DIR="${ROOT_DIR}/patch"

# Problem Size
INPUT_SIZE=8192
INPUT_SIZE_3D=288

# ==========================================
# --- 2. Helper Functions ---
# ==========================================

log()      { echo "[$(date '+%H:%M:%S')] $*"; }
log_warn() { echo "[$(date '+%H:%M:%S')] WARN: $*" >&2; }
log_err()  { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; }

# Native device __float128 is compiled by this project for SM 10.0 and newer.
supports_naive_fp128() {
  local arch_digits="${ARCH%%[!0-9]*}"
  [[ -n "$arch_digits" ]] && ((10#$arch_digits >= 100))
}

# Append a tab-separated result row: Framework  Stencil  Detail  GStPt/s
record_result() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "${RESULTS_FILE}"; }

run_or_die() {
  local msg="$1"; shift
  "$@"
  local status=$?
  if [ $status -ne 0 ]; then
    log_err "$msg (Exit Code: $status)"
    exit $status
  fi
}

setup_repo() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local patch_file="${4:-}"

  if [ ! -d "$name" ]; then
    run_or_die "Failed to clone $name" git clone "$url" "$name"
  fi
  
  pushd "$name" > /dev/null
  git checkout "$commit" &> /dev/null
  if [[ -n "$patch_file" && -f "$patch_file" ]]; then
    git apply --check "$patch_file" &> /dev/null
    if [ $? -eq 0 ]; then
      git apply "$patch_file"
    fi
  fi
  popd > /dev/null
}

# ==========================================
# --- 3. Argument Parsing ---
# ==========================================

print_usage() {
  echo "Usage: $(basename "$0") [options]"
  echo "  -c, --clean         Clean results and benchmark build directories"
  echo "  -h, --help          Show this help"
  echo ""
  echo "  Requires CUDA_COMPUTE_CAPABILITY to be set in the environment"
  echo "  (e.g. export CUDA_COMPUTE_CAPABILITY=120). This is done automatically"
  echo "  when running via entrypoint.sh."
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--clean) CLEAN_FIRST=1; shift ;;
    -h|--help) print_usage; exit 0 ;;
    *) log_warn "Unknown option: $1 (ignored)"; shift ;;
  esac
done


# ==========================================
# --- 4. Benchmark Modules ---
# ==========================================

domain_size_for_stencil() {
  case "$1" in
    3x3x3*|5x5x5*) printf '%sx%sx%s\n' "$INPUT_SIZE_3D" "$INPUT_SIZE_3D" "$INPUT_SIZE_3D" ;;
    *)              printf '%sx%s\n' "$INPUT_SIZE" "$INPUT_SIZE" ;;
  esac
}

bench_naive() {
  log "--- Running naive baselines ---"
  pushd "$ROOT_DIR" > /dev/null

  local exe="./build/bin/bench-stencil"
  local stencil_dir="data/stencils"
  # The m=1, m=2, and m=None stencil variants use the same naive kernel.
  local stencil_names=(
    "7x7LmNone_0"
    "15x15LmNone_0"
    "5x5x5LmNone_0"
  )

  for name in "${stencil_names[@]}"; do
    local stencil_file="${stencil_dir}/${name}.stc"
    local domain_size
    domain_size=$(domain_size_for_stencil "$name")
    local output gstpts_64 gstpts_32
    local naive_algorithms="naive-fp64,naive-fp32"
    if supports_naive_fp128; then
      naive_algorithms+=",naive-fp128"
    fi

    log "Benchmarking naive kernels: ${name}"
    output=$("$exe" -S "$domain_size" --stencil-file "$stencil_file" \
      -A "$naive_algorithms" -T 128 -W "$WARMUP_RUNS" 2>/dev/null)
    gstpts_64=$(awk '/^Naive FP64 GStPt\/s / {print $NF}' <<< "$output")
    gstpts_32=$(awk '/^Naive FP32 GStPt\/s / {print $NF}' <<< "$output")
    record_result "Direct FP64" "$name" "-" "$gstpts_64"
    record_result "Direct FP32" "$name" "-" "$gstpts_32"
    if supports_naive_fp128; then
      local gstpts_128
      gstpts_128=$(awk '/^Naive FP128 GStPt\/s / {print $NF}' <<< "$output")
      record_result "Direct FP128" "$name" "-" "$gstpts_128"
    fi
  done
  popd > /dev/null
}

bench_ozaki() {
  log "--- Running OzakiStencil ---"
  pushd "$ROOT_DIR" > /dev/null

  local exe="./build/bin/bench-stencil"
  local stencil_dir="data/stencils"

  local stencil_names=(
    "7x7Lm1_0"
    "7x7Lm2_0"
    "7x7LmNone_0"
    "15x15Lm1_0"
    "15x15Lm2_0"
    "15x15Lm3_0"
    "15x15LmNone_0"
    "5x5x5Lm1_0"
    "5x5x5Lm2_0"
    "5x5x5LmNone_0"
  )

  # Data slices control input precision.  They are independent of the packed
  # coefficient-kernel variants selected by the stencil files below.
  local slice_counts=(5 9 13 17)
  for name in "${stencil_names[@]}"; do
    local stencil_file="${stencil_dir}/${name}.stc"
    local domain_size
    domain_size=$(domain_size_for_stencil "$name")
    log "Benchmarking OzakiStencil: ${name}"

    for s in "${slice_counts[@]}"; do
      log "  slices=${s}"

      local output gstpts
      output=$("$exe" -S "$domain_size" --stencil-file "$stencil_file" \
        -A ozaki -s "$s" -T 128 -W "$WARMUP_RUNS" 2>/dev/null)
      gstpts=$(awk '/^OzakiStencil GStPt\/s / {print $NF}' <<< "$output")
      record_result "OzakiStencil" "$name" "s=${s}" "$gstpts"
    done
  done
  popd > /dev/null
}

bench_convstencil() {
  log "--- Running ConvStencil ---"
  pushd "$BENCH_DIR" > /dev/null
  setup_repo "ConvStencil" "https://github.com/microsoft/ConvStencil.git" "89688a1" "${PATCH_DIR}/convstencil.patch"
  
  cd ConvStencil
  mkdir -p build && cd build
  run_or_die "Configuring ConvStencil" cmake .. -DGPU_ARCH="${ARCH}"
  run_or_die "Building ConvStencil" make convstencil_2d -j"${JOBS}"
  
  declare -A MAP=( 
    ["box2d1r"]="box2d1r" 
    ["box2d3r"]="box2d3r" 
  )

  for bench_arg in "${!MAP[@]}"; do
    local std_name="${MAP[$bench_arg]}"
    local total_runs=$((128 + WARMUP_RUNS))
    
    log "Benchmarking ConvStencil: ${bench_arg} -> ${std_name}"

    local cmd="./convstencil_2d ${bench_arg} ${INPUT_SIZE} ${INPUT_SIZE} ${total_runs}"
    gstpts=$(eval "$cmd" | awk '/GStencil\/s/ {print $3}')
    record_result "ConvStencil" "$std_name" "-" "$gstpts"
  done
  popd > /dev/null
}

bench_ebisu() {
  log "--- Running EBISU ---"
  pushd "$BENCH_DIR" > /dev/null
  setup_repo "EBISU" "https://github.com/neozhang307/EBISU-ICS23.git" "dd3d844" "${PATCH_DIR}/ebisu.patch"

  cd EBISU/2dstencil

  # Adjust TSTEP for box2d1r based on per-SM shared memory limits.
  local tstep_box2d1r
  case "${ARCH}" in
    90|90a)   tstep_box2d1r=6 ;;
    100|100a) tstep_box2d1r=6 ;;
    120)      tstep_box2d1r=5 ;;
    *)        tstep_box2d1r=5     # conservative default
              log_warn "Unknown arch ${ARCH} for EBISU TSTEP tuning, using TSTEP=${tstep_box2d1r}" ;;
  esac
  sed -i "/timesteps< 1,  box_shape,  4,  double>/,/};/{
    /\/\//! s/val = [0-9]\+/val = ${tstep_box2d1r}/
  }" common/temporalconfig.cuh

  run_or_die "Configuring EBISU" sh config.sh "${ARCH}"
  run_or_die "Building EBISU" sh build.sh

  declare -A MAP=(
    ["2d9pt"]="box2d1r"
    ["2d49pt"]="box2d3r"
  )

  for bench_dir in "${!MAP[@]}"; do
    local std_name="${MAP[$bench_dir]}"
    local total_runs=$((128 + WARMUP_RUNS))
    local exe="./build/init/${bench_dir}/${bench_dir}_ebisu.exe"

    log "Benchmarking EBISU: ${bench_dir} -> ${std_name}"

    local cmd="${exe} ${INPUT_SIZE} ${INPUT_SIZE} --iter=${total_runs} --verbose"
    gstpts=$(eval "$cmd" 2>/dev/null | awk '/GCells\/s/ && !/Valid/ {print $NF}')
    record_result "EBISU" "$std_name" "-" "$gstpts"
  done
  popd > /dev/null
}

# ==========================================
# --- 5. Main Execution Flow ---
# ==========================================

if [ "$CLEAN_FIRST" -eq 1 ]; then
  log "Cleaning results and benchmark build directories..."
  rm -rf "${REPORT_DIR}"
  rm -rf "${BENCH_DIR}/ConvStencil/build"
  rm -rf "${BENCH_DIR}/EBISU/2dstencil/build"
fi

mkdir -p "${REPORT_DIR}"
mkdir -p "${BENCH_DIR}"

RESULTS_FILE="${REPORT_DIR}/results.tsv"
printf 'Framework\tStencil\tDetail\tGStPt/s\n' > "${RESULTS_FILE}"

bench_naive
bench_ozaki
cd "$BENCH_DIR" || exit 1
bench_convstencil
bench_ebisu

log "All benchmarks completed."
log "Results saved to ${RESULTS_FILE}"

log "Generating performance bar chart..."
python3 "${ROOT_DIR}/scripts/plot_perf_bar.py" "${RESULTS_FILE}" \
    -o "${REPORT_DIR}/perf_bar.eps" \
    -g "sm_${ARCH}"

echo ""
echo "================================================================"
echo "  Summary — 2D ${INPUT_SIZE}x${INPUT_SIZE}; 3D ${INPUT_SIZE_3D}x${INPUT_SIZE_3D}x${INPUT_SIZE_3D}, sm_${ARCH}"
echo "================================================================"
awk -F'\t' '
  NR==1 { next }
  { printf "%-20s %-16s %-8s %s\n", $1, $2, $3, $4 }
' "${RESULTS_FILE}"
echo "================================================================"
