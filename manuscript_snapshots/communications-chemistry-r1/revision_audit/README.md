# CANAL: Censored Assay-Noise Activity Learner

CANAL models inequality-qualified, source-heterogeneous bioactivity observations with
molecular features, lightweight assay context, an anchored censored likelihood, and
relation-specific weights. The primary benchmark is ChEMBL 35 top-20 human
single-protein IC50 prediction.

## Evidence status

- S1--S4: 20 targets, four leakage-controlled grouped protocols, five seeds, and 400
  target-split-seed runs per complete method.
- S5: a label-free source-context-tail stress test with 34,563 test records and zero
  direct within-target parent, document, or assay overlap.
- Modern comparators: protocol-matched D-MPNN and AttentiveFP bounded-MSE models.
- External confirmation: a locked ChEMBL 36 update with 69,300 records from 1,112
  documents absent from ChEMBL 35.
- Additional audits: strict matched ablation, exact-record calibration, chemical
  similarity strata, target-wise temporal/source composition, and monitored resource use.

The supported claim is a protocol-specific accuracy-boundary advantage. The results do
not establish universal molecular-property or drug-target-affinity SOTA.

## Repository layout

- `src/canal/`: reusable feature, split, loss, and evaluation code.
- `scripts/`: data construction, training, summarization, and statistical analysis.
- `tests/`: unit tests for censored objectives, chemistry utilities, and Tobit fitting.
- `reports/`: compact machine-readable results and analysis reports.
- `docs/`: resource policy and literature-collision boundary.
- `EXPERIMENT_LOG.md`: chronological experiment record.

Raw ChEMBL files, processed parquet tables, model outputs, logs, and local environments
are intentionally excluded from Git. See `DATA_PROVENANCE.md`.

## Environment and smoke test

```bash
cd /home/wsk/projects/33_CANAL
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
```

Download ChEMBL 35 and build a small relation-preserving dataset:

```bash
bash scripts/download_chembl_v35.sh
python scripts/build_chembl_canal.py --smoke-targets 3 --out-prefix chembl35_smoke
CANAL_CPU_THREADS=32 bash scripts/run_smoke_protocol.sh
```

## Final evidence workflows

The final analyses use seeds `2027 2039 2053 2069 2081`. Important entry points are:

```bash
bash scripts/run_submission_completion.sh
bash scripts/run_chembl36_release_update.sh
bash scripts/run_s5_modern_5seed.sh
python scripts/analyze_matched_ablation.py
python scripts/analyze_calibration.py
python scripts/summarize_similarity_5seed.py
python scripts/summarize_efficiency.py
```

Exact command lines and completed run identifiers are recorded in `EXPERIMENT_LOG.md`.
The resource policy caps a run at four GPUs and 192 CPU threads on the experiment host.
Individual model jobs use much smaller thread counts so the machine retains headroom.

## Result provenance

All numerical comparison tables are produced by this project. Published values are not
mixed into the matched rankings. Confirmatory and exploratory statistical families are
identified in their manifests under `reports/`; S5 neural intervals are explicitly
exploratory. A publication freeze should add checksums for the selected processed data,
prediction exports, and public archive after the final repository URL is assigned.
