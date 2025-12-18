#!/bin/bash

ARCH="80"
declare -A PROBLEM_SIZES
RERUN_NCU=0
PROBLEM_SIZES["L2_0"]=1024
PROBLEM_SIZES["L2_1"]=2048
PROBLEM_SIZES["global_1"]=4096
PROBLEM_SIZES["global_2"]=8192

STENCIL_RADII=(0 1 2 3 4 8 12 20 28 36)

# Create output directory if it doesn't exist to prevent nsys errors
mkdir -p benchmark/reports/bottlenecks

# Initialize CSV with headers
echo "intent,problem_size,radius,run_count,total_ns,mean_ns,median_ns,variance_ns2,spills_bytes" > benchmark/reports/bottlenecks.csv

SQL_QUERY="SELECT (k.end - k.start)
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS s ON k.demangledName = s.id
WHERE k.start NOT IN (
    SELECT start FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start ASC LIMIT 10
);"

for key in "${!PROBLEM_SIZES[@]}"; do
  for size in "${STENCIL_RADII[@]}"; do
    echo "Running test ${key}, ${size}"
    
    report_name="benchmark/reports/bottlenecks/${key}_${size}"
    
    cmd_nsys="python3 scripts/cuda_stencil_box_size.py --arch ${ARCH} -r ${size} -i ${PROBLEM_SIZES[$key]}"
    cmd_ncu="python3 scripts/cuda_stencil_box_size.py --arch ${ARCH} -r ${size} -i ${PROBLEM_SIZES[$key]} -w 0 -n 1"
    
    spills=$(nsys profile --trace="cuda" \
        --stats=true \
        --gpu-metrics-devices=cuda-visible \
        -o "${report_name}" \
        --force-overwrite=true \
        $cmd_nsys | grep -oP 'REG_SPILL=\K\d+')
    echo "  Spilled bytes for (${key}, ${size}) = ${spills}"

    if [ ! -f "${report_name}.ncu-rep" -o "${RERUN_NCU}" -eq 1 ]; then
      echo "  Generating kernel profile with NCU..."
      ncu --set full -o "${report_name}" $cmd_ncu
    fi

    export DB_PATH="${report_name}.sqlite"
    export QUERY="${SQL_QUERY}"
    
    echo "  Extracting information about (${key}, ${size})..."
    row=$(python3 -c "
import sqlite3
import sys
import os
import statistics

try:
    conn = sqlite3.connect(os.environ['DB_PATH'])
    cursor = conn.cursor()
    cursor.execute(os.environ['QUERY'])
    
    # Fetch all durations (tuples of 1 element)
    rows = [r[0] for r in cursor.fetchall()]
    
    if not rows:
        print(',,,,') # Empty result
    else:
        count = len(rows)
        total = sum(rows)
        mean = statistics.mean(rows)
        median = statistics.median(rows)
        variance = statistics.variance(rows) if count > 1 else 0.0
        
        # Print CSV formatted string
        print(f'{count},{total},{mean:.4f},{median:.4f},{variance:.4f}')

except Exception as e:
    print(f'Error processing {os.environ.get(\"DB_PATH\")}: {e}', file=sys.stderr)
    sys.exit(1)
")
    
    if [ $? -eq 0 ]; then
      echo "${key},${PROBLEM_SIZES[$key]},${size},${row},${spills}" >> benchmark/reports/bottlenecks.csv
      echo "  Done with (${key}, ${size})"
    else
      echo "  Failed to extract metrics for (${key}, ${size})"
    fi
  done
done
