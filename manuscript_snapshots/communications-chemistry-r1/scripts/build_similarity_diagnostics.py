#!/usr/bin/env python
"""Build split-aware chemical-similarity diagnostics for CANAL figures.

The prediction export reuses row identifiers across evaluation protocols, so
nearest-training similarity must be keyed by both row_id and split.  This
script computes that quantity from the frozen molecular table, adds
target-split cluster-bootstrap intervals, and audits split overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator


SPLITS = ["S0", "S1", "S2", "S3", "S4"]
ENDPOINTS = ["IC50", "Ki", "EC50", "Kd"]
SIMILARITY_BINS = [-np.inf, 0.3, 0.5, 0.7, np.inf]
SIMILARITY_LABELS = ["<0.3", "0.3-0.5", "0.5-0.7", ">=0.7"]
TARGET_COLUMNS = [
    "target_chembl_id",
    "target_pref_name",
    "parent_inchikey",
    "canonical_smiles_rdkit",
    *[f"split_{split.lower()}" for split in SPLITS],
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def molecule_key(frame: pd.DataFrame) -> pd.Series:
    parent = frame["parent_inchikey"].astype("string")
    fallback = "ROW:" + frame["row_id"].astype(str)
    return parent.where(parent.notna() & parent.ne(""), fallback)


def fingerprint(smiles: object, generator):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    return None if mol is None else generator.GetFingerprint(mol)


def target_similarities(payload: tuple[str, pd.DataFrame]) -> pd.DataFrame:
    target, frame = payload
    frame = frame.copy()
    frame["molecule_key"] = molecule_key(frame)
    molecules = (
        frame.sort_values("row_id")
        .drop_duplicates("molecule_key")[["molecule_key", "canonical_smiles_rdkit"]]
    )
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_by_key = {
        row.molecule_key: fingerprint(row.canonical_smiles_rdkit, generator)
        for row in molecules.itertuples(index=False)
    }

    chunks: list[pd.DataFrame] = []
    for split in SPLITS:
        split_col = f"split_{split.lower()}"
        train_keys = frame.loc[frame[split_col].eq("train"), "molecule_key"].drop_duplicates()
        train_fps = [fp_by_key[key] for key in train_keys if fp_by_key.get(key) is not None]
        if not train_fps:
            raise RuntimeError(f"No valid training fingerprints for {target}-{split}")

        test = frame.loc[
            frame[split_col].eq("test"),
            ["row_id", "target_chembl_id", "target_pref_name", "parent_inchikey", "molecule_key"],
        ].copy()
        test_molecules = test.drop_duplicates("molecule_key")
        similarities: dict[str, float] = {}
        for row in test_molecules.itertuples(index=False):
            fp = fp_by_key.get(row.molecule_key)
            if fp is None:
                similarities[row.molecule_key] = np.nan
                continue
            values = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            similarities[row.molecule_key] = float(max(values))
        test["split"] = split
        test["max_train_tanimoto"] = test["molecule_key"].map(similarities)
        chunks.append(test.drop(columns="molecule_key"))

    result = pd.concat(chunks, ignore_index=True)
    if result["max_train_tanimoto"].isna().any():
        bad = int(result["max_train_tanimoto"].isna().sum())
        raise RuntimeError(f"{target}: {bad} test rows have no valid fingerprint similarity")
    return result


def cluster_bootstrap_interval(
    frame: pd.DataFrame,
    metric: str,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, int]:
    valid = frame.dropna(subset=[metric]).copy()
    grouped = valid.groupby(["target_chembl_id", "split"], sort=True)[metric].agg(["sum", "count"])
    clusters = len(grouped)
    if clusters < 2:
        return np.nan, np.nan, clusters
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, clusters, size=(samples, clusters))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high), clusters


def build_similarity_summary(
    predictions: pd.DataFrame,
    row_similarity: pd.DataFrame,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    pred = predictions.merge(
        row_similarity[["row_id", "split", "max_train_tanimoto"]],
        on=["row_id", "split"],
        how="left",
        validate="many_to_one",
    )
    if pred["max_train_tanimoto"].isna().any():
        raise RuntimeError("Prediction rows are missing split-aware similarity values")
    pred["similarity_bin"] = pd.cut(
        pred["max_train_tanimoto"],
        bins=SIMILARITY_BINS,
        labels=SIMILARITY_LABELS,
        right=False,
    )
    pred["censored_dir_ok"] = pd.to_numeric(pred["censored_dir_ok"], errors="coerce")

    rows: list[dict[str, object]] = []
    grouped = pred.groupby(
        ["similarity_bin", "p_relation", "treatment"],
        observed=True,
        sort=False,
    )
    for group_index, ((sim_bin, relation, treatment), group) in enumerate(grouped):
        exact = group["is_exact"].fillna(False).astype(bool)
        row: dict[str, object] = {
            "similarity_bin": str(sim_bin),
            "relation": relation,
            "treatment": treatment,
            "rows": len(group),
        }
        metric_specs = {
            "mae": (group.loc[exact].copy(), "abs_error"),
            "censored_nll": (group.loc[~exact].copy(), "censored_nll"),
            "censored_dir": (group.loc[~exact].copy(), "censored_dir_ok"),
        }
        cluster_counts = []
        for metric_index, (output_name, (metric_frame, source_name)) in enumerate(metric_specs.items()):
            values = pd.to_numeric(metric_frame[source_name], errors="coerce")
            metric_frame[source_name] = values
            row[output_name] = values.mean()
            low, high, clusters = cluster_bootstrap_interval(
                metric_frame,
                source_name,
                samples=bootstrap_samples,
                seed=bootstrap_seed + 100 * group_index + metric_index,
            )
            row[f"{output_name}_ci_low"] = low
            row[f"{output_name}_ci_high"] = high
            if values.notna().any():
                cluster_counts.append(clusters)
        row["clusters"] = max(cluster_counts, default=0)
        rows.append(row)
    return pd.DataFrame(rows)


def build_applicability_domain(row_similarity: pd.DataFrame) -> pd.DataFrame:
    molecules = row_similarity.copy()
    molecules["molecule_key"] = molecule_key(molecules)
    molecules = molecules.drop_duplicates(["target_chembl_id", "split", "molecule_key"])
    summary = (
        molecules.groupby(
            ["split", "target_chembl_id", "target_pref_name"],
            sort=True,
            dropna=False,
        )["max_train_tanimoto"]
        .agg(test_molecules="size", median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
        .reset_index()
    )
    summary["split"] = pd.Categorical(summary["split"], SPLITS, ordered=True)
    return summary.sort_values(["split", "target_chembl_id"]).reset_index(drop=True)


def build_split_overlap(dataset: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    rows = []
    columns = {
        "molecule": "parent_inchikey",
        "assay": "assay_chembl_id",
        "document": "document_chembl_id",
        "scaffold": "scaffold",
    }
    for split in SPLITS:
        split_col = f"split_{split.lower()}"
        train = dataset[dataset[split_col].eq("train")]
        validation = dataset[dataset[split_col].eq("val")]
        test = dataset[dataset[split_col].eq("test")]
        row: dict[str, object] = {
            "endpoint": endpoint,
            "split": split,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        }
        for label, column in columns.items():
            train_values = set(train[column].dropna().astype(str)) - {""}
            test_values = set(test[column].dropna().astype(str)) - {""}
            row[f"{label}_train_test_overlap"] = len(train_values & test_values)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=360_033)
    args = parser.parse_args()

    repo = args.repo.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = repo / "data" / "processed" / "chembl35_ic50_top20_v2.parquet"
    prediction_path = repo / "results" / "predictions" / "ic50_lgbm_relation_aware_s2027.parquet"

    dataset = pd.read_parquet(dataset_path).reset_index(names="row_id")
    missing = sorted(set(TARGET_COLUMNS) - set(dataset.columns))
    if missing:
        raise RuntimeError(f"Dataset is missing required columns: {missing}")
    target_frame = dataset[["row_id", *TARGET_COLUMNS]].copy()
    target_groups = [(target, group) for target, group in target_frame.groupby("target_chembl_id", sort=True)]

    cpu_count = os.cpu_count() or 1
    worker_limit = max(1, min(cpu_count // 2, cpu_count - 30))
    workers = min(args.workers or worker_limit, worker_limit, len(target_groups))
    print(f"Computing {len(target_groups)} targets with {workers} workers (CPU count={cpu_count})", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        chunks = list(executor.map(target_similarities, target_groups))
    row_similarity = pd.concat(chunks, ignore_index=True).sort_values(["split", "target_chembl_id", "row_id"])
    if row_similarity.duplicated(["row_id", "split"]).any():
        raise RuntimeError("row_id + split is not unique in the similarity table")

    predictions = pd.read_parquet(prediction_path)
    prediction_keys = predictions[["row_id", "split"]].drop_duplicates()
    matched = prediction_keys.merge(
        row_similarity[["row_id", "split"]],
        on=["row_id", "split"],
        how="left",
        indicator=True,
    )
    if not matched["_merge"].eq("both").all():
        raise RuntimeError("Not every frozen prediction row has a split-aware similarity")

    summary = build_similarity_summary(
        predictions,
        row_similarity,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    applicability = build_applicability_domain(row_similarity)
    overlap_frames = []
    for endpoint in ENDPOINTS:
        endpoint_path = (
            repo
            / "data"
            / "processed"
            / f"chembl35_{endpoint.lower()}_top20_v2.parquet"
        )
        endpoint_data = pd.read_parquet(
            endpoint_path,
            columns=[
                "parent_inchikey",
                "assay_chembl_id",
                "document_chembl_id",
                "scaffold",
                *[f"split_{split.lower()}" for split in SPLITS],
            ],
        )
        overlap_frames.append(build_split_overlap(endpoint_data, endpoint))
    overlap = pd.concat(overlap_frames, ignore_index=True)

    paths = {
        "row_similarity": output / "row_similarity_by_split.csv",
        "similarity_summary": output / "similarity_strata_cluster_ci.csv",
        "applicability_domain": output / "applicability_domain_by_target.csv",
        "split_overlap": output / "split_overlap_audit.csv",
    }
    row_similarity.to_csv(paths["row_similarity"], index=False)
    summary.to_csv(paths["similarity_summary"], index=False)
    applicability.to_csv(paths["applicability_domain"], index=False)
    overlap.to_csv(paths["split_overlap"], index=False)

    manifest = {
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "fingerprint": "Morgan radius 2, 2048 bits (ECFP4-equivalent)",
        "similarity": "maximum Tanimoto to same-target training molecules, recomputed per split",
        "molecule_unit": "parent InChIKey; row_id fallback only when parent is missing",
        "bootstrap_unit": "target x split",
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "workers": workers,
        "versions": {
            "rdkit": rdBase.rdkitVersion,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "counts": {
            "row_split_pairs": len(row_similarity),
            "prediction_row_split_pairs": len(prediction_keys),
            "targets": int(row_similarity["target_chembl_id"].nunique()),
            "splits": sorted(row_similarity["split"].unique().tolist()),
        },
    }
    manifest_path = output / "similarity_diagnostics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for name, path in [*paths.items(), ("manifest", manifest_path)]:
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()
