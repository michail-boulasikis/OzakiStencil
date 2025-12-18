#!/bin/bash
set -u # Error on undefined variables

# ==========================================
# --- 1. Configuration & Constants ---
# ==========================================

# Default Arguments
SHUFFLE_SEED=12345
WARMUP_RUNS=10
ARCH="80"       # Default: A100 (sm_80)
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
INPUT_SIZE=8160

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
  echo "  -t, --timeout [int] Timeout per benchmark autotuning step"
  echo "  --arch [int]        SM compute capability (e.g., 80 for A100)"
  echo "  --ncu               Generate NCU reports (script runs slower)"
  echo "  -h, --help          Show this help"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--clean) CLEAN_FIRST=1; shift ;;
    -t|--timeout-per-benchmark) TIMEOUT_PER_BENCH="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    -h|--help) print_usage; exit 0 ;;
    --ncu) USE_NCU=1; shift ;;
    --) shift; CMAKE_ARGS+=("$@"); break ;;
    *) CMAKE_ARGS+=("$1"); shift ;;
  esac
done

SM_ARCH="sm_${ARCH}"

# ==========================================
# --- 4. Benchmark Modules ---
# ==========================================

bench_ozaki() {
  log_info "--- Running OzakiStencil ---"
  run_or_die "OzakiStencil build failed" sh run.sh -- -DCUDA_COMPUTE_CAPABILITY="${ARCH}" "${CMAKE_ARGS[@]}"
}

bench_plain_cuda() {
  log_info "--- Running Plain CUDA ---"
  
  declare -A MAP=( 
    ["box2d1r"]="box2d1r" 
    ["box2d2r"]="box2d2r" 
    ["heat2d1r"]="star2d1r" 
    ["heat2d2r"]="star2d2r" 
  )

  for bench_arg in "${!MAP[@]}"; do
    local std_name="${MAP[$bench_arg]}"
    log_info "Profiling CUDA: $bench_arg -> $std_name"
    
    local cmd="python ${ROOT_DIR}/scripts/cuda_stencil_kernel_gen.py --arch ${ARCH} --stencil ${bench_arg} --input-size ${INPUT_SIZE}"
    run_nsys "${REPORT_DIR}/cuda_${std_name}" $cmd
    run_ncu "${REPORT_DIR}/cuda_${std_name}" $cmd
  done
}

bench_cudnn() {
  log_info "--- Running cuDNN ---"
  
  declare -A MAP=( 
    ["1"]="box2d1r" 
    ["2"]="box2d2r" 
    ["3"]="box2d3r"
  )
  local bin_path="${ROOT_DIR}/build/bin/bench-cudnn"

  for size in "${!MAP[@]}"; do
    local std_name="${MAP[$size]}"
    log_info "Profiling cuDNN: Radius $size -> $std_name"
    
    local cmd="${bin_path} ${size} ${INPUT_SIZE}"
    run_nsys "${REPORT_DIR}/cudnn_${std_name}" $cmd
    run_ncu "${REPORT_DIR}/cudnn_${std_name}" $cmd
  done
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
    ["star2d1r"]="star2d1r" 
    ["star2d3r"]="star2d3r" 
    ["box2d1r"]="box2d1r" 
    ["box2d3r"]="box2d3r" 
  )

  for bench_arg in "${!MAP[@]}"; do
    local std_name="${MAP[$bench_arg]}"
    local total_runs=$((128 + WARMUP_RUNS))
    
    log_info "Profiling ConvStencil: ${bench_arg} -> ${std_name}"

    local cmd="./convstencil_2d ${bench_arg} ${INPUT_SIZE} ${INPUT_SIZE} ${total_runs}"
    run_nsys "${REPORT_DIR}/convstencil_${std_name}" $cmd
    run_ncu "${REPORT_DIR}/convstencil_${std_name}" $cmd
  done
  popd > /dev/null
}

bench_lorastencil() {
  log_info "--- Running LoRAStencil ---"
  pushd "$BENCH_DIR" > /dev/null
  setup_repo "LoRAStencil" "https://github.com/HPHEX/LoRAStencil.git" "1cdb7e2" "${PATCH_DIR}/lorastencil.patch"

  cd LoRAStencil
  mkdir -p build && cd build
  run_or_die "Configuring LoRAStencil" cmake .. -DGPU_ARCH="${ARCH}"
  run_or_die "Building LoRAStencil" make lorastencil_2d -j"${JOBS}"

  declare -A MAP=( 
    ["star2d1r"]="star2d1r" 
    ["star2d3r"]="star2d3r" 
    ["box2d1r"]="box2d1r" 
    ["box2d3r"]="box2d3r" 
  )

  for bench_arg in "${!MAP[@]}"; do
    local std_name="${MAP[$bench_arg]}"
    local total_runs=$((128 + WARMUP_RUNS))
    
    log_info "Profiling LoRAStencil: ${bench_arg} -> ${std_name}"

    local cmd="./lorastencil_2d ${bench_arg} ${INPUT_SIZE} ${INPUT_SIZE} ${total_runs}"
    run_nsys "${REPORT_DIR}/lorastencil_${std_name}" $cmd
    run_ncu "${REPORT_DIR}/lorastencil_${std_name}" $cmd
  done
  popd > /dev/null
}

bench_drstencil() {
  log_info "--- Running DRStencil (Auto-tuning) ---"
  pushd "$BENCH_DIR" > /dev/null
  setup_repo "DRStencil" "https://github.com/simple86/DRStencil.git" "2f95b25" "${PATCH_DIR}/drstencil.patch"
  
  cd DRStencil
  run_or_die "Compiling DRStencil Generator" make -j"${JOBS}"
  cd benchmarks || exit 1

  declare -A MAP=( 
    ["2d5pt_star"]="star2d1r" 
    ["2d9pt_box"]="box2d1r" 
    ["2d25pt_box"]="box2d2r" 
    ["2d9pt_star"]="star2d2r" 
  )

  for benchdir in "${!MAP[@]}"; do
    local std_name="${MAP[$benchdir]}"
    
    if [ ! -d "${benchdir}" ]; then log_warn "DRStencil bench ${benchdir} missing"; continue; fi
    
    pushd "${benchdir}" > /dev/null
    log_info "Tuning ${benchdir} -> ${std_name} (Timeout: ${TIMEOUT_PER_BENCH}s)"

    mkdir -p cu prof bin
    cp ../../drstencil .
    cp common.hpp cu/
    sed -i "s/8192/${INPUT_SIZE}/g" "${benchdir}.stc"

    python3 tuning.py > /dev/null

    local files=(cu/*.cu)
    local shuffled_files=($(shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:"${SHUFFLE_SEED}" -nosalt -md sha256 </dev/zero 2>/dev/null) -e "${files[@]}"))

    local best_time=""
    local best_file=""
    local start_time=$(date +%s)

    for file in "${shuffled_files[@]}"; do
      local current_time=$(date +%s)
      local elapsed=$((current_time - start_time))
      if [ "$elapsed" -ge "$TIMEOUT_PER_BENCH" ]; then
        log_warn "Timeout reached for ${benchdir}"
        break
      fi

      local exename=$(basename "${file}" .cu)
      
      sed -i 's/t < Iterations;/t < 128;/g' "${file}"

      nvcc "${file}" -maxrregcount=128 -arch="${SM_ARCH}" -O3 -o "bin/${exename}" \
        -ccbin=g++ -std=c++11 -Xcompiler "-fPIC -fopenmp -O3 -fno-strict-aliasing" \
        --use_fast_math -Xptxas "-dlcm=cg" &> /dev/null

      run_nsys "prof/${exename}" "bin/${exename}" > /dev/null 2>&1

      local this_time=$(get_kernel_time_from_sqlite "prof/${exename}.sqlite")

      if [[ -n "$this_time" ]]; then
        local is_better=$(python3 -c "import sys; print(1 if '${best_time}' == '' or float('${this_time}') < float('${best_time}') else 0)")
        if [ "$is_better" -eq 1 ]; then
          best_time=$this_time
          best_file="prof/${exename}"
        fi
      fi
    done

    # Save Best
    if [[ -n "$best_file" ]]; then
      local final_nsys="${REPORT_DIR}/drstencil_${std_name}.nsys-rep"
      local final_sql="${REPORT_DIR}/drstencil_${std_name}.sqlite"
      local final_ncu="${REPORT_DIR}/drstencil_${std_name}" # no ext for ncu arg
      
      local best_exename=$(basename "${best_file}") 

      log_info "Winner for ${std_name}: ${best_exename}"
      cp "${best_file}.nsys-rep" "${final_nsys}"
      cp "${best_file}.sqlite" "${final_sql}"
      run_ncu "${final_ncu}" "bin/${best_exename}" > /dev/null 2>&1
      
      echo "${std_name},${best_exename}" >> "${ROOT_DIR}/data/drstencil_best_config.csv"
    else
      log_err "No valid result found for ${benchdir}"
    fi
    popd > /dev/null
  done
  popd > /dev/null
}

bench_tcstencil() {
  log_info "--- Running TCStencil ---"
  pushd "$BENCH_DIR" > /dev/null
  setup_repo "TCStencil" "https://github.com/buaa-hipo/TCStencil.git" "4587a33" "${PATCH_DIR}/tcstencil.patch"
  
  cd TCStencil
  
  local tc_arch="A100"
  case ${ARCH} in
    70) tc_arch="V100" ;;
    80) tc_arch="A100" ;;
    86) tc_arch="A4000" ;;
    90) tc_arch="H100" ;;
    100) tc_arch="B200" ;;
  esac
  python3 script/nvcc_compile.py --arch "${tc_arch}" > /dev/null

  declare -A MAP=( 
    ["star2d1r"]="star2d1r" 
    ["star2d2r"]="star2d2r" 
    ["box2d1r"]="box2d1r" 
    ["box2d2r"]="box2d2r" 
  )
  
  for bench_name in "${!MAP[@]}"; do
    local std_name="${MAP[$bench_name]}"
    local best_time=""
    local best_file=""
    
    shopt -s nullglob
    local files=("bin_ncu/tensor_${bench_name}_${INPUT_SIZE}"*)
    shopt -u nullglob

    if [ ${#files[@]} -eq 0 ]; then
      log_warn "No binaries found for TCStencil ${bench_name}"
      continue
    fi

    log_info "Tuning TCStencil: ${bench_name} -> ${std_name}"

    for file in "${files[@]}"; do
      local base_filename="tc_$(basename "${file}")"
      
      run_nsys "${base_filename}" "${file}" > /dev/null 2>&1
      
      local this_time=$(get_kernel_time_from_sqlite "${base_filename}.sqlite")

      if [[ -n "$this_time" ]]; then
        local is_better=$(python3 -c "print(1 if '${best_time}' == '' or float('${this_time}') < float('${best_time}') else 0)")
        if [ "$is_better" -eq 1 ]; then
          best_time=$this_time
          best_file="${file}"
        fi
      fi
    done

    if [[ -n "$best_file" ]]; then
      log_info "Best TCStencil ${std_name}: ${best_file} ($best_time ns)"
      
      local best_report_base="tc_$(basename "${best_file}")"
      
      cp "${best_report_base}.nsys-rep" "${REPORT_DIR}/tcstencil_${std_name}.nsys-rep"
      cp "${best_report_base}.sqlite" "${REPORT_DIR}/tcstencil_${std_name}.sqlite"
      
      run_ncu "${REPORT_DIR}/tcstencil_${std_name}" "${best_file}" 
    fi
  done
  popd > /dev/null
}

# ==========================================
# --- 5. Main Execution Flow ---
# ==========================================

if [ "$CLEAN_FIRST" -eq 1 ]; then
  log_info "Cleaning directories..."
  rm data/drstencil_best_config.csv
  rm -rf build/ data/reports/
  rm -rf "${BENCH_DIR}/DRStencil"
  rm -rf "${BENCH_DIR}/ConvStencil"
  rm -rf "${BENCH_DIR}/TCStencil"
  rm -rf "${BENCH_DIR}/LoRAStencil"
fi

mkdir -p "${REPORT_DIR}"
mkdir -p "${BENCH_DIR}"

bench_ozaki
cd "$BENCH_DIR" || exit 1
bench_plain_cuda
bench_cudnn
bench_convstencil
bench_lorastencil
bench_drstencil
bench_tcstencil

log_info "All benchmarks completed."
