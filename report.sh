#!/bin/bash -l

QUERY="
SELECT
  SUM(k.end - k.start) as total_ns,
  AVG(k.end - k.start) as mean_ns,
  MEDIAN(k.end - k.start) as median_ns
FROM
  CUPTI_ACTIVITY_KIND_KERNEL AS k 
    JOIN StringIds AS s ON k.demangledName = s.id 
  WHERE k.start NOT IN ( SELECT start FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start ASC LIMIT 10 ) 
    AND s.value NOT LIKE '%gold%'
    GROUP BY s.value;
"

echo "experiment,total_ns,avg_ns,median_ns"
for file in data/reports/H100/*.sqlite; do
  instance=$(basename ${file} ".sqlite")
  row=$(sqlite3 --csv --noheader ${file} "${QUERY}")
  echo "${instance},${row}"
done
