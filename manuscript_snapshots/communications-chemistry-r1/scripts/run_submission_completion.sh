#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_PY="${ENV_PY:-$ROOT/.conda/bin/python}"
THREADS="${CANAL_JOB_THREADS:-8}"
MAX_PARALLEL="${CANAL_MAX_PARALLEL:-4}"
DATASET="data/processed/chembl35_ic50_top20_v2.parquet"
SEEDS=(2027 2039 2053 2069 2081)
SPLITS=(S1 S2 S3 S4)

mkdir -p logs

run_split() {
  local split="$1"
  local tag="submission_tobit_desc_source_light_5seed_${split}"
  CANAL_CPU_THREADS="$THREADS" PYTHONPATH=src "$ENV_PY" scripts/run_tobit_baseline.py \
    --dataset "$DATASET" \
    --splits "$split" \
    --seeds "${SEEDS[@]}" \
    --max-targets 20 \
    --feature-set desc \
    --metadata-mode source_light \
    --sigma-calibration mixed_grid \
    --ridge-alpha 10.0 \
    --max-iter 300 \
    --treatments tobit_exact_only tobit_standard \
    --tag "$tag" \
    > "logs/${tag}.log" 2>&1
}

active=0
for split in "${SPLITS[@]}"; do
  run_split "$split" &
  active=$((active + 1))
  if (( active >= MAX_PARALLEL )); then
    wait -n
    active=$((active - 1))
  fi
done
wait

echo "[done] submission completion Tobit runs finished"
