#!/bin/bash
set -u # Error on undefined variables

# ==========================================
# --- 1. Configuration & Constants ---
# ==========================================

# Default Arguments
SHUFFLE_SEED=12345
WARMUP_RUNS=10
ARCH="120"
TIMEOUT_PER_BENCH=600
CLEAN_FIRST=0
CMAKE_ARGS=()
USE_NCU=0
JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)

# Paths
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="${ROOT_DIR}/benchmark"
REPORT_DIR="${ROOT_DIR}/data/reports"
PATCH_DIR="${ROOT_DIR}/patch"

# Problem Size
INPUT_SIZE=8192

# Database Query
SQL_QUERY="SELECT
  SUM(k.end - k.start)
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s
  ON k.demangledName = s.id
WHERE k.start NOT IN (
  SELECT start
  FROM CUPTI_ACTIVITY_KIND_KERNEL
  ORDER BY start ASC
  LIMIT ${WARMUP_RUNS}
) AND s.value NOT LIKE '%gold%'
GROUP BY s.value;"

# ==========================================
# --- 2. Helper Functions ---
# ==========================================

log_info() { echo -e "\033[0;34m[INFO]\033[0m $1"; }
log_warn() { echo -e "\033[0;33m[WARN]\033[0m $1"; }
log_err()  { echo -e "\033[0;31m[ERROR]\033[0m $1"; }

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

get_kernel_time_from_sqlite() {
  local db_path="$1"
  python3 -c "
import sqlite3, sys
try:
  conn = sqlite3.connect('${db_path}')
  cursor = conn.cursor()
  cursor.execute(\"\"\"${SQL_QUERY}\"\"\")
  result = cursor.fetchone()
  print(result[0] if result and result[0] is not None else '')
except Exception as e:
  sys.exit(1)
"
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

run_nsys() {
  local output_path="$1"
  shift
  if [ ! -f "${output_path}.nsys-rep" ]; then
    log_info "Profiling NSYS: $(basename "${output_path}")"
    nsys profile \
      --trace="cuda,nvtx" \
      --stats=true \
      --gpu-metrics-devices=cuda-visible \
      --force-overwrite=true \
      -o "${output_path}" \
      "$@" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
      log_warn "NSYS exited with error for ${output_path}, result is likely wrong."
    fi
  else
    log_info "Report $(basename "${output_path}") already exists. Skipping."
  fi
}

run_ncu() {
  if [ "${USE_NCU}" -eq 0 ]; then return; fi

  local output_path="$1"
  shift
  if [ ! -f "${output_path}.ncu-rep" ]; then
    log_info "Profiling NCU: $(basename "${output_path}")"
    ncu \
      --set full \
      --launch-skip "${WARMUP_RUNS}" \
      --launch-count 1 \
      --target-processes all \
      --force-overwrite \
      --export "${output_path}" \
      "$@" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
      log_warn "NCU exited with error for ${output_path}, result is likely wrong."
    fi
  else
    log_info "Report $(basename "${output_path}").ncu-rep already exists. Skipping."
  fi
}

# ==========================================
# --- 3. Argument Parsing ---
# ==========================================

print_usage() {
  echo "Usage: $(basename "$0") [options] -- [cmake options]"
  echo "  -c, --clean         Clean repositories and reports"
  echo "  --arch [str]        SM compute capability (e.g., 120 for RTX 5080, 90a for H100/H200)"
  echo "  --ncu               Generate NCU reports (script runs slower)"
  echo "  -h, --help          Show this help"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--clean) CLEAN_FIRST=1; shift ;;
    --arch) ARCH="$2"; shift 2 ;; # e.g. 120 for RTX 5080, 90a for H100/H200
    -h|--help) print_usage; exit 0 ;;
    --ncu) USE_NCU=1; shift ;;
    --) shift; CMAKE_ARGS+=("$@"); break ;;
    *) CMAKE_ARGS+=("$1"); shift ;;
  esac
done


# ==========================================
# --- 4. Benchmark Modules ---
# ==========================================

bench_ozaki() {
  log_info "--- Running OzakiStencil ---"
  pushd "$ROOT_DIR" > /dev/null
  run_or_die "OzakiStencil build failed" sh run.sh -t -- -DCUDA_COMPUTE_CAPABILITY="${ARCH}" "${CMAKE_ARGS[@]}"

  local exe="./build/bin/bench-stencil"
  local stencil_dir="data/stencils"

  declare -A STENCILS=(
    ["7x7LmNone_0"]="${stencil_dir}/7x7LmNone_0.stc"
    ["7x7Lm1_0"]="${stencil_dir}/7x7Lm1_0.stc"
    ["7x7Lm2_0"]="${stencil_dir}/7x7Lm2_0.stc"
  )

  local slice_counts=(5 7 10 13 17)
  local profile_slices=10  # only generate NSYS/NCU reports for this slice count

  for name in "${!STENCILS[@]}"; do
    local stencil_file="${STENCILS[$name]}"
    local base_args="-S ${INPUT_SIZE}x${INPUT_SIZE} --stencil-file ${stencil_file}"

    log_info "Benchmarking OzakiStencil: ${name}"

    # FP64 baseline (independent of slice count, run once)
    local output gstpts_64 gstpts_32
    output=$($exe $base_args -A naive-fp64,naive-fp32 -T 128 -W ${WARMUP_RUNS} -R 2 2>/dev/null)
    gstpts_64=$(echo "$output" | awk '/^Naive FP64/ {print $NF}')
    gstpts_32=$(echo "$output" | awk '/^Naive FP32/ {print $NF}')
    record_result "Direct FP64" "$name" "-" "$gstpts_64"
    record_result "Direct FP32" "$name" "-" "$gstpts_32"

    for s in "${slice_counts[@]}"; do
      log_info "  slices=${s}"

      output=$($exe $base_args -A convstencil -s "$s" -T 128 -W ${WARMUP_RUNS} -R 2 2>/dev/null)
      gstpts=$(echo "$output" | awk '/^Ozaki ConvStencil/ {print $NF}')
      record_result "OzakiConvStencil" "$name" "s=${s}" "$gstpts"

      # NSYS/NCU reports only for the representative slice count
      if [ "$s" -eq "${profile_slices}" ]; then
        run_nsys "${REPORT_DIR}/ozaki_naive_fp64_${name}" \
          $exe $base_args -A naive-fp64 -T 128 -W ${WARMUP_RUNS} -R 1
        run_nsys "${REPORT_DIR}/ozaki_conv_${name}" \
          $exe $base_args -A convstencil -s "$s" -T 128 -W ${WARMUP_RUNS} -R 1

        if [ "${USE_NCU}" -eq 1 ]; then
          for ncu_spec in "naive-fp64:ozaki_naive_fp64" "convstencil:ozaki_conv"; do
            local algo="${ncu_spec%%:*}"
            local report="${ncu_spec##*:}_${name}"
            local ncu_args="$base_args -A ${algo} --ncu-mode"
            [ "$algo" = "convstencil" ] && ncu_args="$ncu_args -s $s"
            if [ ! -f "${REPORT_DIR}/${report}.ncu-rep" ]; then
              log_info "Profiling NCU: ${report}"
              ncu \
                --set full \
                --target-processes all \
                --force-overwrite \
                --export "${REPORT_DIR}/${report}" \
                $exe $ncu_args > /dev/null 2>&1
              if [ $? -ne 0 ]; then
                log_warn "NCU exited with error for ${report}, result is likely wrong."
              fi
            else
              log_info "Report ${report}.ncu-rep already exists. Skipping."
            fi
          done
        fi
      fi
    done
  done
  popd > /dev/null
}

bench_convstencil() {
  log_info "--- Running ConvStencil ---"
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
    
    log_info "Profiling ConvStencil: ${bench_arg} -> ${std_name}"

    local cmd="./convstencil_2d ${bench_arg} ${INPUT_SIZE} ${INPUT_SIZE} ${total_runs}"
    gstpts=$(eval "$cmd" | awk '/GStencil\/s/ {print $3}')
    record_result "ConvStencil" "$std_name" "-" "$gstpts"
    run_nsys "${REPORT_DIR}/convstencil_${std_name}" $cmd
    run_ncu "${REPORT_DIR}/convstencil_${std_name}" $cmd
  done
  popd > /dev/null
}

bench_ebisu() {
  log_info "--- Running EBISU ---"
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

    log_info "Profiling EBISU: ${bench_dir} -> ${std_name}"

    local cmd="${exe} ${INPUT_SIZE} ${INPUT_SIZE} --iter=${total_runs} --verbose"
    gstpts=$(eval "$cmd" 2>/dev/null | awk '/GCells\/s/ && !/Valid/ {print $NF}')
    record_result "EBISU" "$std_name" "-" "$gstpts"
    run_nsys "${REPORT_DIR}/ebisu_${std_name}" $cmd
    run_ncu "${REPORT_DIR}/ebisu_${std_name}" $cmd
  done
  popd > /dev/null
}

bench_spider() {
  log_info "--- Running SPIDER (FP16) ---"
  pushd "$BENCH_DIR" > /dev/null
  setup_repo "SPIDER" "https://github.com/KevinWu2017/SPIDER.git" "8cb9a83" "${PATCH_DIR}/spider.patch"

  cd SPIDER
  git submodule update --init third_party/argparse

  mkdir -p build && cd build
  run_or_die "Configuring SPIDER" cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_ARCH="${ARCH}"
  run_or_die "Building SPIDER" cmake --build . --config Release -j"${JOBS}"
  cd ..

  # SPIDER only implements box2d7r (15x15 box, radius 7) with FP16 sparse tensor cores.
  # GStencil/s is divided by 4 to normalize FP16 -> FP64 (sizeof ratio 8/2 = 4).
  local std_name="box2d7r"
  local total_runs=$((128 + WARMUP_RUNS))
  local exe="./build/bin/stencil_2d_half_sparse"

  log_info "Profiling SPIDER: stencil_2d_half_sparse -> ${std_name}"

  local cmd="${exe} -M ${INPUT_SIZE} -N ${INPUT_SIZE} -T ${total_runs} --profile"
  echo "SPIDER GStPt/s (${std_name}) = $(eval "$cmd" 2>/dev/null | awk -F', ' '{printf "%.2f", $NF / 4}')"
  run_nsys "${REPORT_DIR}/spider_${std_name}" $cmd
  run_ncu "${REPORT_DIR}/spider_${std_name}" $cmd
  popd > /dev/null
}

# ==========================================
# --- 5. Main Execution Flow ---
# ==========================================

if [ "$CLEAN_FIRST" -eq 1 ]; then
  log_info "Cleaning directories..."
  rm -rf build/ data/reports/
  rm -rf "${BENCH_DIR}/ConvStencil"
  rm -rf "${BENCH_DIR}/EBISU"
  rm -rf "${BENCH_DIR}/SPIDER"
fi

mkdir -p "${REPORT_DIR}"
mkdir -p "${BENCH_DIR}"

RESULTS_FILE="${REPORT_DIR}/results.tsv"
printf 'Framework\tStencil\tDetail\tGStPt/s\n' > "${RESULTS_FILE}"

bench_ozaki
cd "$BENCH_DIR" || exit 1
bench_convstencil
bench_ebisu

log_info "All benchmarks completed."
log_info "Results saved to ${RESULTS_FILE}"

echo ""
echo "================================================================"
echo "  Summary — ${INPUT_SIZE}x${INPUT_SIZE}, sm_${ARCH}"
echo "================================================================"
awk -F'\t' '
  NR==1 { next }
  { printf "%-20s %-16s %-8s %s\n", $1, $2, $3, $4 }
' "${RESULTS_FILE}"
echo "================================================================"
