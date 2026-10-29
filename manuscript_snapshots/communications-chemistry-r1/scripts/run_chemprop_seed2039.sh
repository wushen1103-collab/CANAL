#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/test/kkkk/33_CANAL
PY="$ROOT/.conda-modern/bin/python"
RUNNER="$ROOT/scripts/run_chemprop_multitask.py"
DATA="$ROOT/data/processed/chembl35_ic50_top20_v2.parquet"
PREPARED="$ROOT/data/processed/chemprop_multitask_ic50"
LOG_DIR="$ROOT/logs/chemprop_modern_src5"
STATUS_DIR="$ROOT/results/modern/chemprop_status"
TAG=modern_src5
SEED=2039
SPLITS=(S1 S2 S3 S4)
GPUS=(0 2 3 4)
TREATMENTS=(exact_only censored_as_exact bounded_mse)

mkdir -p "$LOG_DIR" "$STATUS_DIR"

run_split() {
  local split=$1
  local gpu=$2
  local failures=0
  for treatment in "${TREATMENTS[@]}"; do
    local stem="chemprop_${split,,}_${treatment}_s${SEED}_${TAG}"
    echo "[$(date -Is)] gpu=$gpu start $stem"
    if "$PY" "$RUNNER" \
      --dataset "$DATA" \
      --prepared-dir "$PREPARED" \
      --split "$split" \
      --seed "$SEED" \
      --treatment "$treatment" \
      --gpu "$gpu" \
      --epochs 30 \
      --patience 6 \
      --batch-size 512 \
      --num-workers 4 \
      --tag "$TAG" >"$LOG_DIR/${stem}.log" 2>&1; then
      printf 'ok\n' >"$STATUS_DIR/${stem}.status"
      echo "[$(date -Is)] gpu=$gpu done $stem"
    else
      printf 'failed\n' >"$STATUS_DIR/${stem}.status"
      echo "[$(date -Is)] gpu=$gpu FAILED $stem"
      failures=$((failures + 1))
    fi
  done
  return "$failures"
}

pids=()
for index in "${!SPLITS[@]}"; do
  run_split "${SPLITS[$index]}" "${GPUS[$index]}" &
  pids+=("$!")
done

exit_code=0
for pid in "${pids[@]}"; do
  wait "$pid" || exit_code=1
done

exit "$exit_code"
