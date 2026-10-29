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
EPOCHS=30
PATIENCE=6
BATCH_SIZE=512
WORKERS=4
GPUS=(0 2 3 4)
SEEDS=(2027 2041 2053 2069 2081)
SPLITS=(S1 S2 S3 S4)
TREATMENTS=(exact_only censored_as_exact bounded_mse)

mkdir -p "$LOG_DIR" "$STATUS_DIR"

run_worker() {
  local gpu=$1
  local offset=$2
  local task_index=0
  local failures=0
  for split in "${SPLITS[@]}"; do
    for treatment in "${TREATMENTS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        if (( task_index % ${#GPUS[@]} == offset )); then
          local stem="chemprop_${split,,}_${treatment}_s${seed}_${TAG}"
          echo "[$(date -Is)] gpu=$gpu start $stem"
          if "$PY" "$RUNNER" \
            --dataset "$DATA" \
            --prepared-dir "$PREPARED" \
            --split "$split" \
            --seed "$seed" \
            --treatment "$treatment" \
            --gpu "$gpu" \
            --epochs "$EPOCHS" \
            --patience "$PATIENCE" \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$WORKERS" \
            --tag "$TAG" >"$LOG_DIR/${stem}.log" 2>&1; then
            printf 'ok\n' >"$STATUS_DIR/${stem}.status"
            echo "[$(date -Is)] gpu=$gpu done $stem"
          else
            printf 'failed\n' >"$STATUS_DIR/${stem}.status"
            echo "[$(date -Is)] gpu=$gpu FAILED $stem"
            failures=$((failures + 1))
          fi
        fi
        task_index=$((task_index + 1))
      done
    done
  done
  return "$failures"
}

pids=()
for offset in "${!GPUS[@]}"; do
  run_worker "${GPUS[$offset]}" "$offset" &
  pids+=("$!")
done

exit_code=0
for pid in "${pids[@]}"; do
  wait "$pid" || exit_code=1
done

ok_count=$(grep -l '^ok$' "$STATUS_DIR"/*_${TAG}.status 2>/dev/null | wc -l)
failed_count=$(grep -l '^failed$' "$STATUS_DIR"/*_${TAG}.status 2>/dev/null | wc -l)
echo "[$(date -Is)] complete ok=$ok_count failed=$failed_count expected=60"
exit "$exit_code"
