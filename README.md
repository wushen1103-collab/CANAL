# CANAL

CANAL is an observation-aware benchmark and calibration framework for molecular bioactivity data containing exact and inequality-qualified measurements. Exact observations contribute Gaussian density terms; lower- and upper-censored observations contribute probability mass over the supported half-line.

The release contains the complete data-construction, benchmark, calibration, statistical-analysis, and external-independence code used for the reported experiments. A compact result snapshot is included so that headline values can be checked without redistributing the source databases or large feature caches.

## Dataset

The complete, frozen version 1.0.0 dataset is permanently deposited in Zenodo at https://doi.org/10.5281/zenodo.22739896. The single downloadable archive contains the processed ChEMBL and BindingDB record tables, 4,200,200 frozen split-role assignments, task and provenance manifests, data dictionary, integrity checks, checksums, and reference benchmark outputs. No registration or access approval is required.

## What is included

- Deterministic construction of the 20-task ChEMBL 37 development cohort.
- Deterministic construction of the 20-task BindingDB 202608 article-derived cohort.
- Random, scaffold, assay-source, temporal, and joint temporal-scaffold splits with five fixed seeds.
- Fifteen classical, tree, censored-neural, SMILES-CNN, and calibrated method variants.
- Validation-only CANAL calibration, target-wise sensitivity analysis, and calibration diagnostics.
- Task-level paired inference, ranking-reversal analysis, semi-synthetic recovery, and metadata-permutation controls.
- Four-layer ChEMBL-BindingDB overlap audit and strict external-cohort construction.
- Compact tables for all 1,350 aggregate benchmark records and the 175-record strict external grid.

## Manuscript snapshots

The source used for the revised Communications Chemistry submission is fixed under
[`manuscript_snapshots/communications-chemistry-r1/`](manuscript_snapshots/communications-chemistry-r1/)
and identified by the Git tag `communications-chemistry-r1`. This snapshot is separate
from the ChEMBL 37 and BindingDB dataset release described below.

## Quick verification

Create an environment and verify the distributed result snapshot:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

An equivalent Conda specification is provided in `environment.yml`; the reference interpreter and complete pinned package list are recorded in `documentation/software_environment.txt`.

## Full reproduction

Download and extract ChEMBL 37 SQLite, and download the BindingDB article-derived TSV archive. The exact source URLs and source-file hashes are recorded in [`manifests/cohorts.json`](manifests/cohorts.json). Then run:

```bash
python scripts/reproduce.py \
  --stage all \
  --chembl-sqlite /path/to/chembl_37.db \
  --bindingdb-zip /path/to/BindingDB_BindingDB_Articles_202608_tsv.zip \
  --device cuda:0 \
  --workers 16
```

Use `--device cpu` on a machine without CUDA. Individual stages (`prepare`, `benchmark`, `diagnostics`, and `external`) can be executed separately. `--dry-run` prints every command without launching an experiment. The `--skip-smiles` option is intended only for development because it omits one of the 15 reported method variants.

Detailed source-data, protocol, output, and resource notes are in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Repository layout

```text
scripts/                 data preparation, training, calibration, and audits
manifests/               source checksums, task lists, task encoding, split hashes
results/benchmark/       complete compact benchmark and calibration results
results/external_strict/ strict BindingDB replication results
results/audits/          provenance overlap and run-accounting outputs
documentation/           reference software environment
```

Large raw databases, derived Parquet files, feature caches, model states, and logs are intentionally excluded from the Git tree. The frozen processed record tables and split assignments are distributed in the Zenodo dataset; source databases and feature caches can also be regenerated with the scripts and are covered by `.gitignore`.

## Primary metric definitions

- **Exact-label MAE:** absolute p-activity error on exact test observations.
- **Censored NLL:** Gaussian density for exact observations and Gaussian half-line probability for bounded observations.
- **Spearman:** computed within each target-endpoint task on exact test observations, then macro-averaged across tasks.
- **Paired inference:** seeds are averaged within task; the 20 target-endpoint tasks are the independent paired units.

CANAL calibration is fitted on validation observations only. Test outcomes, censoring relations, and thresholds enter after model and calibration parameters are frozen, solely for evaluation.

## Data terms

This repository distributes code, task manifests, and compact derived statistics. The frozen processed dataset is available from Zenodo, while the ChEMBL and BindingDB source databases remain governed by their respective providers' terms and should be obtained from the official download pages.

## License

The code is released under the MIT License. Third-party data and dependencies retain their own licenses and terms.
