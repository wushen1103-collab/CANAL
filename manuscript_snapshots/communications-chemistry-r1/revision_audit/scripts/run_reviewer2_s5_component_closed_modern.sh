#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/wsk/projects/33_CANAL
PY="$ROOT/.conda-modern/bin/python"
DATA="$ROOT/data/processed/reviewer2_quality/chembl35_ic50_top20_v2_s5_component_closed.parquet"
PREPARED="$ROOT/data/processed/chemprop_multitask_ic50_s5_component_closed"
GRAPHS="$ROOT/data/processed/attentivefp_graphs_ic50"
TAG=r2_s5_component_closed_5seed

run_seed() {
    local gpu="$1"
    local seed="$2"
    "$PY" "$ROOT/scripts/run_chemprop_multitask.py" \
        --dataset "$DATA" --prepared-dir "$PREPARED" --split S5 \
        --seed "$seed" --treatment bounded_mse --gpu "$gpu" \
        --epochs 30 --patience 6 --batch-size 512 --num-workers 4 --tag "$TAG" \
        >"$ROOT/logs/reviewer2/chemprop_s5_component_closed_s${seed}.log" 2>&1
    "$PY" "$ROOT/scripts/run_attentivefp_multitask.py" \
        --dataset "$DATA" --prepared-dir "$PREPARED" --graph-cache "$GRAPHS" \
        --split S5 --seed "$seed" --treatment bounded_mse --gpu "$gpu" \
        --epochs 30 --patience 6 --batch-size 512 --num-workers 4 --tag "$TAG" \
        >"$ROOT/logs/reviewer2/attentivefp_s5_component_closed_s${seed}.log" 2>&1
}

mkdir -p "$ROOT/logs/reviewer2"
(run_seed 0 2027; run_seed 0 2081) &
pid0=$!
run_seed 1 2039 &
pid1=$!
run_seed 2 2053 &
pid2=$!
run_seed 3 2069 &
pid3=$!
wait "$pid0" "$pid1" "$pid2" "$pid3"
touch "$ROOT/reports/reviewer2_validation_2026-10-28/s5_component_closed_modern.done"
