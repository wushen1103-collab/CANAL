#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CANAL_CPU_THREADS="${CANAL_CPU_THREADS:-32}"
if [ "$CANAL_CPU_THREADS" -gt 192 ]; then
  CANAL_CPU_THREADS=192
fi
export CANAL_CPU_THREADS
export OMP_NUM_THREADS="$CANAL_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CANAL_CPU_THREADS"
export MKL_NUM_THREADS="$CANAL_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CANAL_CPU_THREADS"

echo "[resource] CANAL_CPU_THREADS=$CANAL_CPU_THREADS (cap=192)"
echo "[resource] CANAL GPU cap is 4; this smoke protocol is CPU-only."

. /home/test/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/.conda"

if ! find data/raw -name 'chembl_35.db' | grep -q .; then
  bash scripts/download_chembl_v35.sh
fi

python scripts/build_chembl_canal.py --smoke-targets 3 --out-prefix chembl35_smoke
python scripts/run_problem_check.py \
  --dataset data/processed/chembl35_smoke.parquet \
  --splits S1 S2 S3 \
  --endpoints IC50 Ki \
  --seeds 2027 2039 2053 \
  --n-jobs "$CANAL_CPU_THREADS" \
  --max-targets 3
python scripts/analyze_results.py --results-dir results/problem_check --out reports/problem_check_summary.csv
