#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.conda/bin/python}"
SEEDS=(2069 2081)
N_JOBS="${CANAL_JOB_THREADS:-12}"
MAX_PARALLEL="${CANAL_MAX_PARALLEL:-4}"
LOG_DIR="$ROOT/logs/5seed_completion_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "[resource] seeds=${SEEDS[*]} n_jobs=${N_JOBS} max_parallel=${MAX_PARALLEL} cpu_thread_cap=192"
echo "[logs] $LOG_DIR"

running=0

wait_for_slot() {
  while (( running >= MAX_PARALLEL )); do
    if wait -n; then
      :
    else
      echo "[warn] one background task failed; continuing to collect other tasks"
    fi
    running=$((running - 1))
  done
}

launch() {
  local name="$1"
  local expected="$2"
  shift 2
  if [[ -s "$expected" ]]; then
    echo "[skip] $name -> $expected"
    return
  fi
  wait_for_slot
  echo "[launch] $name"
  (
    set -x
    "$@"
  ) >"$LOG_DIR/${name}.log" 2>&1 &
  running=$((running + 1))
}

pc_expected() {
  local dataset="$1"
  local model="$2"
  local tag="$3"
  echo "$ROOT/results/problem_check/problem_check_${dataset}_${model}_${tag}.csv"
}

common_pc=(
  --splits S1 S2 S3 S4
  --seeds "${SEEDS[@]}"
  --max-targets 20
  --n-jobs "$N_JOBS"
  --sigma-calibration mixed_grid
)

ic50_dataset="data/processed/chembl35_ic50_top20_v2.parquet"

launch xgb_plain_ecfp_cw0p1_aw1 \
  "$(pc_expected chembl35_ic50_top20_v2 xgb 5seed_add_xgb_plain_ecfp_cw0p1_aw1_s2069_2081)" \
  "$PY" scripts/run_problem_check.py --dataset "$ic50_dataset" "${common_pc[@]}" \
  --endpoints IC50 --model xgb --feature-set ecfp --metadata-mode none \
  --treatments exact_only censored_as_exact censored_likelihood \
  --censor-weight 0.1 --censored-anchor-weight 1.0 \
  --tag 5seed_add_xgb_plain_ecfp_cw0p1_aw1_s2069_2081

launch xgb_source_light_ecfp_cw0p1_aw1 \
  "$(pc_expected chembl35_ic50_top20_v2 xgb 5seed_add_xgb_source_light_ecfp_cw0p1_aw1_s2069_2081)" \
  "$PY" scripts/run_problem_check.py --dataset "$ic50_dataset" "${common_pc[@]}" \
  --endpoints IC50 --model xgb --feature-set ecfp --metadata-mode source_light \
  --treatments exact_only censored_as_exact censored_likelihood \
  --censor-weight 0.1 --censored-anchor-weight 1.0 \
  --tag 5seed_add_xgb_source_light_ecfp_cw0p1_aw1_s2069_2081

launch xgb_source_context_ecfp_cw0p1_aw1 \
  "$(pc_expected chembl35_ic50_top20_v2 xgb 5seed_add_xgb_source_context_ecfp_cw0p1_aw1_s2069_2081)" \
  "$PY" scripts/run_problem_check.py --dataset "$ic50_dataset" "${common_pc[@]}" \
  --endpoints IC50 --model xgb --feature-set ecfp --metadata-mode source_context \
  --treatments exact_only censored_as_exact censored_likelihood \
  --censor-weight 0.1 --censored-anchor-weight 1.0 \
  --tag 5seed_add_xgb_source_context_ecfp_cw0p1_aw1_s2069_2081

launch xgb_source_quality_ecfp_cw0p1_aw1 \
  "$(pc_expected chembl35_ic50_top20_v2 xgb 5seed_add_xgb_source_quality_ecfp_cw0p1_aw1_s2069_2081)" \
  "$PY" scripts/run_problem_check.py --dataset "$ic50_dataset" "${common_pc[@]}" \
  --endpoints IC50 --model xgb --feature-set ecfp --metadata-mode source_quality \
  --treatments exact_only censored_as_exact censored_likelihood \
  --censor-weight 0.1 --censored-anchor-weight 1.0 \
  --tag 5seed_add_xgb_source_quality_ecfp_cw0p1_aw1_s2069_2081

launch xgb_source_light_ecfp_desc_cw0p1_aw1 \
  "$(pc_expected chembl35_ic50_top20_v2 xgb 5seed_add_xgb_source_light_ecfp_desc_cw0p1_aw1_s2069_2081)" \
  "$PY" scripts/run_problem_check.py --dataset "$ic50_dataset" "${common_pc[@]}" \
  --endpoints IC50 --model xgb --feature-set ecfp_desc --metadata-mode source_light \
  --treatments exact_only censored_as_exact censored_likelihood \
  --censor-weight 0.1 --censored-anchor-weight 1.0 \
  --tag 5seed_add_xgb_source_light_ecfp_desc_cw0p1_aw1_s2069_2081

launch rf_source_light_ecfp \
  "$(pc_expected chembl35_ic50_top20_v2 rf 5seed_add_rf_source_light_ecfp_s2069_2081)" \
  "$PY" scripts/run_problem_check.py --dataset "$ic50_dataset" "${common_pc[@]}" \
  --endpoints IC50 --model rf --feature-set ecfp --metadata-mode source_light \
  --treatments exact_only censored_as_exact \
  --tag 5seed_add_rf_source_light_ecfp_s2069_2081

launch ridge_source_light_ecfp_desc \
  "$(pc_expected chembl35_ic50_top20_v2 ridge 5seed_add_ridge_source_light_ecfp_desc_s2069_2081)" \
  "$PY" scripts/run_problem_check.py --dataset "$ic50_dataset" "${common_pc[@]}" \
  --endpoints IC50 --model ridge --feature-set ecfp_desc --metadata-mode source_light \
  --treatments exact_only censored_as_exact \
  --tag 5seed_add_ridge_source_light_ecfp_desc_s2069_2081

launch lgbm_source_light_ecfp_cw0p1_aw1 \
  "$(pc_expected chembl35_ic50_top20_v2 lgbm 5seed_add_lgbm_source_light_ecfp_cw0p1_aw1_s2069_2081)" \
  "$PY" scripts/run_problem_check.py --dataset "$ic50_dataset" "${common_pc[@]}" \
  --endpoints IC50 --model lgbm --feature-set ecfp --metadata-mode source_light \
  --treatments exact_only censored_as_exact censored_likelihood \
  --censor-weight 0.1 --censored-anchor-weight 1.0 \
  --tag 5seed_add_lgbm_source_light_ecfp_cw0p1_aw1_s2069_2081

launch lgbm_source_light_ecfp_desc_cw0p1_aw1 \
  "$(pc_expected chembl35_ic50_top20_v2 lgbm 5seed_add_lgbm_source_light_ecfp_desc_cw0p1_aw1_s2069_2081)" \
  "$PY" scripts/run_problem_check.py --dataset "$ic50_dataset" "${common_pc[@]}" \
  --endpoints IC50 --model lgbm --feature-set ecfp_desc --metadata-mode source_light \
  --treatments exact_only censored_as_exact censored_likelihood \
  --censor-weight 0.1 --censored-anchor-weight 1.0 \
  --tag 5seed_add_lgbm_source_light_ecfp_desc_cw0p1_aw1_s2069_2081

for spec in \
  "cw0p25_aw0p5 0.25 0.5" \
  "cw0p25_aw1 0.25 1.0" \
  "cw0p5_aw0p5 0.5 0.5" \
  "cw0p5_aw1 0.5 1.0" \
  "pure_censored 1.0 0.0"; do
  read -r label cw aw <<<"$spec"
  launch "lgbm_ecfp_desc_${label}" \
    "$(pc_expected chembl35_ic50_top20_v2 lgbm 5seed_add_lgbm_ecfp_desc_${label}_s2069_2081)" \
    "$PY" scripts/run_problem_check.py --dataset "$ic50_dataset" "${common_pc[@]}" \
    --endpoints IC50 --model lgbm --feature-set ecfp_desc --metadata-mode source_light \
    --treatments censored_likelihood \
    --censor-weight "$cw" --censored-anchor-weight "$aw" \
    --tag "5seed_add_lgbm_ecfp_desc_${label}_s2069_2081"
done

launch lgbm_ecfp_desc_relation_cw0p25_aw1_right0p1_aw0 \
  "$(pc_expected chembl35_ic50_top20_v2 lgbm 5seed_add_lgbm_ecfp_desc_relation_cw0p25_aw1_right0p1_aw0_s2069_2081)" \
  "$PY" scripts/run_problem_check.py --dataset "$ic50_dataset" "${common_pc[@]}" \
  --endpoints IC50 --model lgbm --feature-set ecfp_desc --metadata-mode source_light \
  --treatments censored_likelihood \
  --censor-weight 0.25 --censored-anchor-weight 1.0 \
  --right-censor-weight 0.1 --right-censored-anchor-weight 0.0 \
  --tag 5seed_add_lgbm_ecfp_desc_relation_cw0p25_aw1_right0p1_aw0_s2069_2081

for split in S1 S2 S3 S4; do
  launch "catboost_source_light_ecfp_desc_${split}" \
    "$(pc_expected chembl35_ic50_top20_v2 catboost 5seed_add_catboost_source_light_ecfp_desc_${split}_s2069_2081)" \
    "$PY" scripts/run_problem_check.py --dataset "$ic50_dataset" \
    --splits "$split" --seeds "${SEEDS[@]}" --max-targets 20 --n-jobs "$N_JOBS" \
    --n-estimators 80 --sigma-calibration mixed_grid \
    --endpoints IC50 --model catboost --feature-set ecfp_desc --metadata-mode source_light \
    --treatments exact_only censored_as_exact \
    --tag "5seed_add_catboost_source_light_ecfp_desc_${split}_s2069_2081"
done

for endpoint in ki ec50 kd; do
  dataset="data/processed/chembl35_${endpoint}_top20_v2.parquet"
  upper="$(echo "$endpoint" | tr '[:lower:]' '[:upper:]')"
  launch "endpoint_${endpoint}_xgb_ecfp_desc_source_light" \
    "$(pc_expected chembl35_${endpoint}_top20_v2 xgb 5seed_add_${endpoint}_xgb_ecfp_desc_source_light_s2069_2081)" \
    "$PY" scripts/run_problem_check.py --dataset "$dataset" "${common_pc[@]}" \
    --endpoints "$upper" --model xgb --feature-set ecfp_desc --metadata-mode source_light \
    --treatments exact_only censored_as_exact censored_likelihood \
    --censor-weight 0.1 --censored-anchor-weight 1.0 \
    --tag "5seed_add_${endpoint}_xgb_ecfp_desc_source_light_s2069_2081"
done

launch tdc_external_dti_fast_core \
  "$ROOT/results/tdc_dti/tdc_dti_external_dti_fast_core_add_s2069_2081.csv" \
  "$PY" scripts/run_tdc_dti_baselines.py \
  --datasets DAVIS KIBA BindingDB_Kd \
  --splits random cold_both \
  --seeds "${SEEDS[@]}" \
  --models ridge xgb lgbm \
  --n-jobs "$N_JOBS" \
  --n-estimators 250 \
  --tag external_dti_fast_core_add_s2069_2081

while (( running > 0 )); do
  if wait -n; then
    :
  else
    echo "[warn] one background task failed during final wait"
  fi
  running=$((running - 1))
done

echo "[done] 5-seed addition jobs finished. Logs: $LOG_DIR"
