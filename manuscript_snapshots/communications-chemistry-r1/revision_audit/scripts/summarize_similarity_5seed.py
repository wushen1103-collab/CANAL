#!/usr/bin/env python
"""Build five-seed, target-balanced similarity-stratified diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BINS = [-np.inf, 0.3, 0.5, 0.7, np.inf]
LABELS = ["<0.3", "0.3-0.5", "0.5-0.7", ">=0.7"]
BOOTSTRAP_SEED = 360_233
BOOTSTRAP_REPLICATES = 20_000


def target_bootstrap(values: pd.Series, rng: np.random.Generator) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if len(array) < 2:
        return np.nan, np.nan
    draws = rng.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))
    means = array[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--row-similarity", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evidence_completion_2026-09-03/similarity_5seed"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat([pd.read_parquet(path) for path in args.predictions], ignore_index=True)
    if predictions["seed"].nunique() != 5:
        raise RuntimeError(f"Expected five seeds, observed {predictions['seed'].nunique()}")
    similarity = pd.read_csv(args.row_similarity)
    similarity = similarity[["row_id", "split", "max_train_tanimoto"]].drop_duplicates()
    if similarity.duplicated(["row_id", "split"]).any():
        raise RuntimeError("row_id + split must be unique in the similarity table")
    merged = predictions.merge(similarity, on=["row_id", "split"], how="left", validate="many_to_one")
    if merged["max_train_tanimoto"].isna().any():
        raise RuntimeError("Missing similarity for one or more prediction rows")
    merged["similarity_bin"] = pd.cut(
        merged["max_train_tanimoto"], BINS, labels=LABELS, right=False
    ).astype("string")
    merged["censored_dir_ok"] = pd.to_numeric(merged["censored_dir_ok"], errors="coerce")
    specs = {
        "=": [("mae", "abs_error")],
        "<": [("censored_nll", "censored_nll"), ("censored_dir", "censored_dir_ok")],
        ">": [("censored_nll", "censored_nll"), ("censored_dir", "censored_dir_ok")],
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    cell_rows = []
    for relation, metric_specs in specs.items():
        relation_frame = merged.loc[merged["p_relation"].eq(relation)].copy()
        for (similarity_bin, treatment), group in relation_frame.groupby(
            ["similarity_bin", "treatment"], sort=False, dropna=False
        ):
            for metric, column in metric_specs:
                valid = group.dropna(subset=[column]).copy()
                cells = (
                    valid.groupby(["target_chembl_id", "split", "seed"], sort=True)[column]
                    .agg(["mean", "size"])
                    .reset_index()
                )
                cells["similarity_bin"] = similarity_bin
                cells["relation"] = relation
                cells["treatment"] = treatment
                cells["metric"] = metric
                cell_rows.append(cells)
                target_means = cells.groupby("target_chembl_id")["mean"].mean()
                low, high = target_bootstrap(target_means, rng)
                rows.append(
                    {
                        "similarity_bin": similarity_bin,
                        "relation": relation,
                        "treatment": treatment,
                        "metric": metric,
                        "target_balanced_mean": target_means.mean(),
                        "ci_low": low,
                        "ci_high": high,
                        "pooled_row_mean": valid[column].mean(),
                        "prediction_rows": len(valid),
                        "unique_test_rows": valid[["row_id", "split"]].drop_duplicates().shape[0],
                        "target_split_seed_cells": len(cells),
                        "targets": len(target_means),
                        "seeds": valid["seed"].nunique(),
                    }
                )
    summary = pd.DataFrame(rows)
    cells = pd.concat(cell_rows, ignore_index=True)
    summary.to_csv(output.with_name(output.name + "_summary.csv"), index=False)
    cells.to_csv(output.with_name(output.name + "_target_split_seed.csv"), index=False)
    manifest = {
        "predictions": [str(path.resolve()) for path in args.predictions],
        "row_similarity": str(args.row_similarity.resolve()),
        "similarity_bins": LABELS,
        "experimental_unit": "target after averaging split and seed",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    output.with_name(output.name + "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(output.with_name(output.name + "_summary.csv"))


if __name__ == "__main__":
    main()

