#!/usr/bin/env python
"""Summarize auxiliary endpoints on protocol-matched method intersections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["endpoint", "target", "split", "seed"]
TREATMENTS = ["exact_only", "censored_as_exact", "censored_likelihood"]
METRICS = ["test_mae", "test_censored_nll", "test_censored_dir_acc", "worst_assay_mae"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    chunks = []
    for path in args.files:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        chunks.append(frame)
    raw = pd.concat(chunks, ignore_index=True)
    selected = raw.loc[
        raw["status"].eq("ok")
        & raw["model"].eq("xgb")
        & raw["feature_set"].eq("ecfp_desc")
        & raw["metadata_mode"].eq("source_light")
        & raw["treatment"].isin(TREATMENTS)
    ].copy()
    duplicate = selected.duplicated(KEYS + ["treatment"], keep=False)
    if duplicate.any():
        columns = KEYS + ["treatment", "source_file"]
        raise ValueError("Duplicate endpoint cells:\n" + selected.loc[duplicate, columns].to_string(index=False))

    coverage = (
        selected.groupby(["endpoint", "treatment"], as_index=False)
        .agg(
            cells=("seed", "size"),
            targets=("target", "nunique"),
            splits=("split", "nunique"),
            seeds=("seed", "nunique"),
        )
        .sort_values(["endpoint", "treatment"])
    )
    coverage.to_csv(args.output / "r2_10e_endpoint_cell_coverage.csv", index=False)

    availability = selected.assign(available=1).pivot_table(
        index=KEYS, columns="treatment", values="available", aggfunc="max", fill_value=0
    )
    for treatment in TREATMENTS:
        if treatment not in availability:
            availability[treatment] = 0
    availability = availability.reset_index()
    availability["all_three"] = availability[TREATMENTS].eq(1).all(axis=1)
    availability["matched_pair"] = availability[
        ["censored_as_exact", "censored_likelihood"]
    ].eq(1).all(axis=1)
    availability.to_csv(args.output / "r2_10e_endpoint_common_cells.csv", index=False)

    target_rows = []
    effect_rows = []
    for endpoint, endpoint_frame in selected.groupby("endpoint"):
        pivot = endpoint_frame.pivot(index=KEYS, columns="treatment", values=METRICS)
        for comparator in ["censored_as_exact", "exact_only"]:
            for metric in METRICS:
                if (metric, "censored_likelihood") not in pivot or (metric, comparator) not in pivot:
                    continue
                pair = pd.concat(
                    [
                        pivot[(metric, "censored_likelihood")].rename("canal"),
                        pivot[(metric, comparator)].rename("comparator"),
                    ],
                    axis=1,
                ).dropna()
                pair = pair.reset_index()
                pair["delta_canal_minus_comparator"] = pair["canal"] - pair["comparator"]
                by_target = (
                    pair.groupby("target", as_index=False)
                    .agg(
                        cells=("seed", "size"),
                        canal=("canal", "mean"),
                        comparator=("comparator", "mean"),
                        delta_canal_minus_comparator=("delta_canal_minus_comparator", "mean"),
                    )
                )
                by_target.insert(0, "metric", metric)
                by_target.insert(0, "comparator_treatment", comparator)
                by_target.insert(0, "endpoint", endpoint)
                target_rows.append(by_target)
                lower_is_better = metric != "test_censored_dir_acc"
                favorable = (
                    by_target["delta_canal_minus_comparator"].le(0)
                    if lower_is_better
                    else by_target["delta_canal_minus_comparator"].ge(0)
                )
                effect_rows.append(
                    {
                        "endpoint": endpoint,
                        "comparator_treatment": comparator,
                        "metric": metric,
                        "paired_cells": len(pair),
                        "paired_targets": by_target["target"].nunique(),
                        "canal_target_balanced_mean": by_target["canal"].mean(),
                        "comparator_target_balanced_mean": by_target["comparator"].mean(),
                        "delta_canal_minus_comparator": by_target[
                            "delta_canal_minus_comparator"
                        ].mean(),
                        "delta_target_sd": by_target["delta_canal_minus_comparator"].std(ddof=1),
                        "favorable_targets": int(favorable.sum()),
                    }
                )

    targets = pd.concat(target_rows, ignore_index=True)
    effects = pd.DataFrame(effect_rows).sort_values(["endpoint", "comparator_treatment", "metric"])
    targets.to_csv(args.output / "r2_10e_endpoint_target_metrics.csv", index=False)
    effects.to_csv(args.output / "r2_10e_endpoint_paired_effects.csv", index=False)

    common_all_three = availability.loc[availability["all_three"], KEYS]
    s3 = selected.loc[selected["split"].eq("S3")].merge(
        common_all_three, on=KEYS, how="inner", validate="many_to_one"
    )
    s3_target = (
        s3.groupby(["endpoint", "treatment", "target"], as_index=False)
        .agg(cells=("seed", "size"), **{metric: (metric, "mean") for metric in METRICS})
    )
    s3_summary = (
        s3_target.groupby(["endpoint", "treatment"], as_index=False)
        .agg(
            targets=("target", "nunique"),
            cells=("cells", "sum"),
            **{metric: (metric, "mean") for metric in METRICS},
        )
    )
    s3_target.to_csv(args.output / "r2_11_s3_auxiliary_target_metrics.csv", index=False)
    s3_summary.to_csv(args.output / "r2_11_s3_auxiliary_summary.csv", index=False)

    manifest = {
        "comparison_unit": "endpoint x target x split x seed",
        "aggregation": "average paired cells within target, then equal-weight targets",
        "canal_treatment": "censored_likelihood",
        "matched_comparator": "censored_as_exact",
        "backbone": "XGBoost for all auxiliary endpoint rows",
        "files": [{"path": str(path), "sha256": sha256(path)} for path in args.files],
        "selected_rows": len(selected),
        "effects": len(effects),
        "s3_common_cells": len(s3),
    }
    (args.output / "r2_10e_endpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(effects.to_string(index=False))


if __name__ == "__main__":
    main()
