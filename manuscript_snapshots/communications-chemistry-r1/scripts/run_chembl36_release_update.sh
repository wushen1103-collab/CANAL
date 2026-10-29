#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.conda/bin/python}"
N_JOBS="${CANAL_JOB_THREADS:-20}"
MAX_PARALLEL="${CANAL_MAX_PARALLEL:-8}"
SEEDS=(2027 2039 2053 2069 2081)
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/chembl36_release_update_$STAMP"
mkdir -p "$LOG_DIR"

if (( N_JOBS * MAX_PARALLEL > 192 )); then
  echo "Refusing to exceed the 192-thread half-host cap" >&2
  exit 2
fi

BATCH_1=(CHEMBL1163125 CHEMBL1865 CHEMBL1974 CHEMBL203 CHEMBL2039)
BATCH_2=(CHEMBL220 CHEMBL230 CHEMBL240 CHEMBL260 CHEMBL279)
BATCH_3=(CHEMBL2835 CHEMBL289 CHEMBL2971 CHEMBL325 CHEMBL3397)
BATCH_4=(CHEMBL340 CHEMBL4005 CHEMBL4296 CHEMBL4822 CHEMBL5145)

echo "[resource] max_parallel=$MAX_PARALLEL n_jobs=$N_JOBS cpu_cap=192 gpu_jobs=0"
echo "[logs] $LOG_DIR"

pids=()
names=()
launch_batch() {
  local cohort="$1"
  local batch="$2"
  shift 2
  local dataset="data/processed/chembl36_ic50_release_update${cohort}.parquet"
  local tag="chembl36_update${cohort}_batch${batch}_locked5"
  local output="results/problem_check/problem_check_$(basename "$dataset" .parquet)_lgbm_${tag}.csv"
  if [[ -s "$output" ]]; then
    echo "[skip] $tag"
    return
  fi
  echo "[launch] $tag targets=$*"
  (
    export OMP_NUM_THREADS="$N_JOBS"
    export OPENBLAS_NUM_THREADS="$N_JOBS"
    export MKL_NUM_THREADS="$N_JOBS"
    "$PY" scripts/run_problem_check.py \
      --dataset "$dataset" \
      --splits UPDATE \
      --endpoints IC50 \
      --seeds "${SEEDS[@]}" \
      --target-ids "$@" \
      --n-jobs "$N_JOBS" \
      --n-estimators 300 \
      --model lgbm \
      --feature-set ecfp_desc \
      --metadata-mode source_light \
      --sigma-calibration mixed_grid \
      --treatments exact_only censored_as_exact censored_likelihood \
      --censor-weight 0.25 \
      --censored-anchor-weight 1.0 \
      --right-censor-weight 0.1 \
      --right-censored-anchor-weight 0.0 \
      --tag "$tag"
  ) >"$LOG_DIR/$tag.log" 2>&1 &
  pids+=("$!")
  names+=("$tag")
}

for cohort in "" "_novel_parent"; do
  launch_batch "$cohort" 1 "${BATCH_1[@]}"
  launch_batch "$cohort" 2 "${BATCH_2[@]}"
  launch_batch "$cohort" 3 "${BATCH_3[@]}"
  launch_batch "$cohort" 4 "${BATCH_4[@]}"
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[done] ${names[$i]}"
  else
    echo "[failed] ${names[$i]}" >&2
    failed=1
  fi
done

echo "$LOG_DIR" > results/problem_check/chembl36_release_update_latest_logdir.txt
exit "$failed"
