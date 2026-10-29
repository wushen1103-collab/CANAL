#!/usr/bin/env python3
"""Assemble complete cross-model worst-assay details for quality ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEYS = ["quality_filter", "endpoint", "target", "split", "seed", "treatment"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-dir", type=Path, required=True)
    args = parser.parse_args()
    quality_dir = args.quality_dir.resolve()
    pairs = pd.read_csv(quality_dir / "r2_9_quality_ablation_cell_pairs.csv")
    eligible = pairs.loc[
        pairs["worst_assay_id_source"].notna()
        & pairs["worst_assay_id_nosource"].notna()
    ].copy()

    same = eligible.loc[eligible["worst_assay_id_agrees"]].copy()
    same_details = same[KEYS].copy()
    same_details["assay_id"] = same["worst_assay_id_source"]
    same_details["is_source_worst"] = True
    same_details["is_nosource_worst"] = True
    same_details["exact_test_n"] = same[
        ["worst_assay_exact_n_source", "worst_assay_exact_n_nosource"]
    ].max(axis=1).astype(int)
    same_details["source_mae"] = same["worst_assay_mae_source"]
    same_details["nosource_mae"] = same["worst_assay_mae_nosource"]
    same_details["source_minus_nosource"] = (
        same_details["source_mae"] - same_details["nosource_mae"]
    )

    mismatched_chunks = []
    for path in sorted((quality_dir / "assay_profiles").glob("*_cross_model_assay_details.csv")):
        mismatched_chunks.append(pd.read_csv(path))
    if not mismatched_chunks:
        raise FileNotFoundError("No mismatched-assay profile outputs found")
    mismatched = pd.concat(mismatched_chunks, ignore_index=True)
    details = pd.concat([same_details, mismatched[same_details.columns]], ignore_index=True)
    details = details.sort_values(KEYS + ["assay_id"]).reset_index(drop=True)

    if details["exact_test_n"].min() < 10:
        raise ValueError("An exported worst-assay detail has fewer than 10 exact test records")
    drivers = details.groupby(KEYS).agg(
        source_driver=("is_source_worst", "max"),
        nosource_driver=("is_nosource_worst", "max"),
    )
    if not drivers[["source_driver", "nosource_driver"]].all().all():
        raise ValueError("At least one eligible cell is missing a model-specific driver assay")
    expected_cells = eligible[KEYS].drop_duplicates()
    observed_cells = details[KEYS].drop_duplicates()
    if len(expected_cells) != len(observed_cells):
        raise ValueError(
            f"Expected {len(expected_cells)} eligible cells, assembled {len(observed_cells)}"
        )

    details.to_csv(quality_dir / "r2_9_worst_assay_cross_model_complete.csv", index=False)
    summary = (
        details.groupby("quality_filter", as_index=False)
        .agg(
            eligible_cells=("seed", lambda values: 0),
            driver_assay_rows=("assay_id", "size"),
            minimum_exact_test_n=("exact_test_n", "min"),
        )
    )
    cell_counts = details[KEYS].drop_duplicates().groupby("quality_filter").size()
    summary["eligible_cells"] = summary["quality_filter"].map(cell_counts).astype(int)
    summary.to_csv(quality_dir / "r2_9_worst_assay_cross_model_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
