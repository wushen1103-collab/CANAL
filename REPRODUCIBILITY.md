# Reproducibility guide

## 1. Reference software environment

The reported runs used Python 3.10.12 with the exact package versions in `requirements.txt`. GPU acceleration is optional for the censored multilayer perceptron and SMILES-CNN routes; the tree and linear routes run on CPU.

For a clean environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch installations can be replaced by the platform-specific wheel recommended for the local CUDA runtime while retaining the recorded PyTorch version.

## 2. Source data

The experiment uses two official releases:

1. ChEMBL 37 SQLite: `chembl_37_sqlite.tar.gz` from the ChEMBL release archive.
2. BindingDB 202608 article-derived TSV: `BindingDB_BindingDB_Articles_202608_tsv.zip` from BindingDB.

The official URLs and expected SHA-256 hashes are stored in `manifests/cohorts.json`. The ChEMBL hash applies to the extracted `chembl_37.db`; the BindingDB hash applies to the downloaded ZIP archive.

Data preparation is deterministic:

```bash
python scripts/reproduce.py \
  --stage prepare \
  --chembl-sqlite /path/to/chembl_37.db \
  --bindingdb-zip /path/to/BindingDB_BindingDB_Articles_202608_tsv.zip \
  --workers 16
```

This creates the ChEMBL development cohort, BindingDB-All, BindingDB-ArticleIndependent, and BindingDB-IndependentStrict under `data/processed/`. Raw and processed databases are ignored by Git.

The selected task tables are frozen in `manifests/chembl37_selected_tasks.csv` and `manifests/bindingdb_202608_selected_tasks.csv`. Gzip-compressed candidate tables expose the complete selection pool. No model result is used during task selection.

## 3. Split protocol

The five seeds are 13, 29, 47, 61, and 83. Each method receives identical row roles for the random, scaffold, source, temporal, and joint protocols. Every benchmark run writes `split_roles_<protocol>_seed<seed>.csv`. SHA-256 hashes of the 25 ChEMBL role files from the reported run are frozen in `manifests/split_role_checksums.csv`.

Temporal and joint roles are deterministic once the cohort is fixed, so their role hashes are identical across seeds. Seed variation remains active for estimator initialization and, where applicable, random role allocation.

## 4. Benchmark execution

Run the complete ChEMBL and BindingDB-All benchmark:

```bash
python scripts/reproduce.py --stage benchmark --device cuda:0 --workers 16
```

The complete benchmark combines:

- core linear and direct censored-likelihood routes;
- tree-ensemble routes;
- validation-calibrated ExtraTrees and histogram-gradient-boosting routes;
- the censored SMILES-CNN;
- semi-synthetic counterparts;
- the metadata-permutation control;
- BindingDB-All core and calibrated sensitivity runs.

The resulting `artifacts/combined/metrics_long.csv` contains 1,350 aggregate records. The primary real-data ChEMBL grid contains 15 methods x 5 protocols x 5 seeds = 375 records.

## 5. Calibration and statistical analysis

```bash
python scripts/reproduce.py --stage diagnostics --workers 16
```

The primary ExtraTrees backbone is fitted on training rows. The positive affine coefficient, additive bias, and residual scale are fitted on validation observations. Test labels are used only after these quantities are fixed.

`analyze_paired_calibration.py` recomputes per-target metrics and uses 20 target-endpoint tasks as paired units after averaging seeds within task. Aggregate inference also averages split-specific values within task. `analyze_targetwise_calibration.py` fits 20 x 5 x 5 validation-only calibrators for sensitivity analysis.

## 6. Strict external replication

```bash
python scripts/reproduce.py --stage external --device cuda:0 --workers 16
```

BindingDB-IndependentStrict removes the union of shared article identifiers, exact molecule-target-endpoint keys, canonical molecule-target pairs, and near-matched measurements within 0.05 p-activity units. The final cohort contains 16,639 records. The strict benchmark contains 7 methods x 5 protocols x 5 seeds = 175 aggregate records.

## 7. Distributed compact results

The `results/` directory contains CSV and JSON outputs needed to verify the reported tables without downloading raw databases. The row-level BindingDB overlap audit and full candidate-task tables are gzip compressed. No model checkpoints, molecular feature caches, raw data, or execution logs are committed.

Run `python scripts/verify_release.py` to check grid coverage, target-level paired units, Spearman invariance, one-hot task identities, overlap counts, and file checksums.
