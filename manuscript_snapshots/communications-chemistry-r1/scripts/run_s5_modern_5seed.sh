#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/test/kkkk/33_CANAL
PY="$ROOT/.conda-modern/bin/python"
DATA="$ROOT/data/processed/chembl35_ic50_top20_v2_s5.parquet"
PREPARED="$ROOT/data/processed/chemprop_multitask_ic50_s5"
GRAPHS="$ROOT/data/processed/attentivefp_graphs_ic50"
TAG=s5_source5_20260903

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

run_seed() {
    local gpu="$1"
    local seed="$2"
    "$PY" "$ROOT/scripts/run_chemprop_multitask.py" \
        --dataset "$DATA" --prepared-dir "$PREPARED" --split S5 \
        --seed "$seed" --treatment bounded_mse --gpu "$gpu" \
        --epochs 30 --patience 6 --batch-size 512 --num-workers 4 --tag "$TAG" \
        >"$ROOT/logs/chemprop_s5_bounded_mse_s${seed}_${TAG}.log" 2>&1
    "$PY" "$ROOT/scripts/run_attentivefp_multitask.py" \
        --dataset "$DATA" --prepared-dir "$PREPARED" --graph-cache "$GRAPHS" \
        --split S5 --seed "$seed" --treatment bounded_mse --gpu "$gpu" \
        --epochs 30 --patience 6 --batch-size 512 --num-workers 4 --tag "$TAG" \
        >"$ROOT/logs/attentivefp_s5_bounded_mse_s${seed}_${TAG}.log" 2>&1
}

mkdir -p "$ROOT/logs" "$ROOT/reports/evidence_completion_2026-09-03"
(run_seed 1 2027; run_seed 1 2081) &
pid1=$!
run_seed 2 2039 &
pid2=$!
run_seed 3 2053 &
pid3=$!
run_seed 4 2069 &
pid4=$!
wait "$pid1" "$pid2" "$pid3" "$pid4"
touch "$ROOT/reports/evidence_completion_2026-09-03/s5_modern_5seed.done"

