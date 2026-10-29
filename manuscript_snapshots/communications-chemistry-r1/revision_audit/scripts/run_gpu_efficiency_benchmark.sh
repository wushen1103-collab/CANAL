#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {dmpnn|attentivefp} GPU_INDEX" >&2
  exit 2
fi

MODEL=$1
GPU=$2
ROOT=/home/wsk/projects/33_CANAL
PY="$ROOT/.conda-modern/bin/python"
TAG="efficiency_monitored_20260903"
LOG="$ROOT/logs/efficiency_${MODEL}_s1_s2027_monitored_20260903.log"
TIME_REPORT="$ROOT/reports/timing_efficiency_${MODEL}_s1_s2027_monitored_20260903.txt"
GPU_LOG="$ROOT/reports/gpu_efficiency_${MODEL}_s1_s2027_monitored_20260903.csv"
SUMMARY="$ROOT/reports/gpu_efficiency_${MODEL}_s1_s2027_monitored_20260903.txt"

mkdir -p "$ROOT/logs" "$ROOT/reports"
printf 'timestamp,used_memory_mib\n' >"$GPU_LOG"

if [[ "$MODEL" == "dmpnn" ]]; then
  COMMAND=(
    "$PY" "$ROOT/scripts/run_chemprop_multitask.py"
    --dataset "$ROOT/data/processed/chembl35_ic50_top20_v2.parquet"
    --prepared-dir "$ROOT/data/processed/chemprop_multitask_ic50"
    --split S1 --seed 2027 --treatment bounded_mse --gpu "$GPU"
    --epochs 30 --patience 6 --batch-size 512 --num-workers 4 --tag "$TAG"
  )
elif [[ "$MODEL" == "attentivefp" ]]; then
  COMMAND=(
    "$PY" "$ROOT/scripts/run_attentivefp_multitask.py"
    --dataset "$ROOT/data/processed/chembl35_ic50_top20_v2.parquet"
    --prepared-dir "$ROOT/data/processed/chemprop_multitask_ic50"
    --graph-cache "$ROOT/data/processed/attentivefp_graphs_ic50"
    --split S1 --seed 2027 --treatment bounded_mse --gpu "$GPU"
    --epochs 30 --patience 6 --batch-size 512 --num-workers 4 --tag "$TAG"
  )
else
  echo "unknown model: $MODEL" >&2
  exit 2
fi

/usr/bin/time -v -o "$TIME_REPORT" "${COMMAND[@]}" >"$LOG" 2>&1 &
COMMAND_PID=$!
while kill -0 "$COMMAND_PID" 2>/dev/null; do
  value=$(nvidia-smi -i "$GPU" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF {sum += $1} END {print sum + 0}')
  printf '%s,%s\n' "$(date -Is)" "$value" >>"$GPU_LOG"
  sleep 1
done
wait "$COMMAND_PID"

peak=$(awk -F, 'NR > 1 && $2 + 0 > max {max = $2 + 0} END {print max + 0}' "$GPU_LOG")
printf 'model=%s\ngpu=%s\npeak_gpu_memory_mib=%s\nsamples=%s\n' \
  "$MODEL" "$GPU" "$peak" "$(( $(wc -l <"$GPU_LOG") - 1 ))" >"$SUMMARY"
cat "$SUMMARY"

