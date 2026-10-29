#!/usr/bin/env python3
"""Audit S1-S4 partition sizes for each ChEMBL 35 endpoint panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


DATASETS = {
    "IC50": "chembl35_ic50_top20_v2.parquet",
    "Ki": "chembl35_ki_top20_v2.parquet",
    "EC50": "chembl35_ec50_top20_v2.parquet",
    "Kd": "chembl35_kd_top20_v2.parquet",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/reviewer2_validation_2026-10-28/split_size_audit"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    inputs: list[dict[str, str]] = []
    for endpoint, filename in DATASETS.items():
        path = root / "data" / "processed" / filename
        frame = pd.read_parquet(path)
        inputs.append({"endpoint": endpoint, "path": str(path), "sha256": sha256(path)})
        for target, target_frame in frame.groupby("target_chembl_id", sort=True):
            for split in ("S1", "S2", "S3", "S4"):
                column = f"split_{split.lower()}"
                if column not in target_frame:
                    continue
                record: dict[str, object] = {
                    "endpoint": endpoint,
                    "target": target,
                    "split": split,
                    "total_rows": int(len(target_frame)),
                    "split_varies_by_seed": False,
                }
                for partition in ("train", "val", "test"):
                    subset = target_frame.loc[target_frame[column].eq(partition)]
                    record[f"{partition}_rows"] = int(len(subset))
                    record[f"{partition}_exact"] = int(subset["p_relation"].eq("=").sum())
                    record[f"{partition}_left"] = int(subset["p_relation"].eq("<").sum())
                    record[f"{partition}_right"] = int(subset["p_relation"].eq(">").sum())
                record["test_fraction"] = record["test_rows"] / record["total_rows"]
                rows.append(record)

    counts = pd.DataFrame(rows)
    counts.to_csv(output_dir / "r2_11_endpoint_target_split_counts.csv", index=False)
    summary = (
        counts.groupby(["endpoint", "split"], as_index=False)
        .agg(
            targets=("target", "nunique"),
            train_rows=("train_rows", "sum"),
            val_rows=("val_rows", "sum"),
            test_rows=("test_rows", "sum"),
            test_exact=("test_exact", "sum"),
            test_left=("test_left", "sum"),
            test_right=("test_right", "sum"),
            median_target_test_fraction=("test_fraction", "median"),
        )
    )
    summary.to_csv(output_dir / "r2_11_endpoint_split_summary.csv", index=False)
    manifest = {
        "splits_are_dataset_fixed_across_model_seeds": True,
        "seed_role": "The five model seeds vary learner stochasticity; they do not regenerate S1-S4 partitions.",
        "inputs": inputs,
        "rows": int(len(counts)),
    }
    (output_dir / "r2_11_split_size_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
