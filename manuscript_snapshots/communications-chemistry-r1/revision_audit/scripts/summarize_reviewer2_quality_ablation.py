#!/usr/bin/env python3
"""Summarize paired source-context ablations on quality-filtered cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


FILTERS = (
    "no_potential_duplicate",
    "no_validity_comment",
    "no_duplicate_or_validity_comment",
    "no_outside_typical_range",
    "no_potential_transcription_error",
)
KEYS = ["endpoint", "target", "split", "seed", "treatment"]
METRICS = ("test_mae", "worst_assay_mae")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 20000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def load_run(path: Path, expected_mode: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame) != 400:
        raise ValueError(f"{path.name}: expected 400 rows, found {len(frame)}")
    if not frame["status"].eq("ok").all():
        bad = frame.loc[~frame["status"].eq("ok"), KEYS + ["status"]]
        raise ValueError(f"{path.name}: failed rows\n{bad.to_string(index=False)}")
    if set(frame["metadata_mode"]) != {expected_mode}:
        raise ValueError(f"{path.name}: unexpected metadata_mode values")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{path.name}: duplicate pairing keys")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--input-dir", type=Path, default=Path("results/problem_check")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/reviewer2_validation_2026-10-28/quality_ablation"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    input_dir = root / args.input_dir
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_frames: list[pd.DataFrame] = []
    input_records: list[dict[str, object]] = []
    for quality_filter in FILTERS:
        source_path = next(input_dir.glob(f"*{quality_filter}_lgbm_r2_{quality_filter}_source_5seed.csv"))
        nosource_path = next(input_dir.glob(f"*{quality_filter}_lgbm_r2_{quality_filter}_nosource_5seed.csv"))
        source = load_run(source_path, "source_light")
        nosource = load_run(nosource_path, "none")

        keep = KEYS + list(METRICS) + [
            "worst_assay_id",
            "worst_assay_exact_n",
            "test_n",
            "test_censored_n",
        ]
        paired = source[keep].merge(
            nosource[keep], on=KEYS, how="outer", validate="one_to_one",
            suffixes=("_source", "_nosource"), indicator=True,
        )
        if not paired["_merge"].eq("both").all():
            raise ValueError(f"{quality_filter}: incomplete source/no-source pairing")
        paired = paired.drop(columns="_merge")
        paired.insert(0, "quality_filter", quality_filter)
        for metric in METRICS:
            paired[f"{metric}_source_minus_nosource"] = (
                paired[f"{metric}_source"] - paired[f"{metric}_nosource"]
            )
        paired["worst_assay_id_agrees"] = (
            paired["worst_assay_id_source"].astype(str)
            == paired["worst_assay_id_nosource"].astype(str)
        )
        pair_frames.append(paired)
        for role, path in (("source", source_path), ("nosource", nosource_path)):
            input_records.append(
                {
                    "quality_filter": quality_filter,
                    "role": role,
                    "path": str(path),
                    "sha256": sha256(path),
                }
            )

    pairs = pd.concat(pair_frames, ignore_index=True)
    pairs.to_csv(output_dir / "r2_9_quality_ablation_cell_pairs.csv", index=False)

    target_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for filter_index, (quality_filter, group) in enumerate(pairs.groupby("quality_filter", sort=True)):
        for metric_index, metric in enumerate(METRICS):
            target = (
                group.groupby("target", as_index=False)
                .agg(
                    source=(f"{metric}_source", "mean"),
                    nosource=(f"{metric}_nosource", "mean"),
                    paired_cells=(metric, "size") if metric in group else (f"{metric}_source", "size"),
                )
            )
            target["source_minus_nosource"] = target["source"] - target["nosource"]
            target.insert(0, "metric", metric)
            target.insert(0, "quality_filter", quality_filter)
            target_rows.extend(target.to_dict("records"))

            diffs = target["source_minus_nosource"].to_numpy()
            ci_low, ci_high = bootstrap_ci(
                diffs, seed=20261028 + filter_index * 10 + metric_index
            )
            try:
                p_value = float(wilcoxon(diffs, alternative="two-sided").pvalue)
            except ValueError:
                p_value = 1.0
            summary_rows.append(
                {
                    "quality_filter": quality_filter,
                    "metric": metric,
                    "source": float(target["source"].mean()),
                    "nosource": float(target["nosource"].mean()),
                    "source_minus_nosource": float(diffs.mean()),
                    "bootstrap_ci_low": float(ci_low),
                    "bootstrap_ci_high": float(ci_high),
                    "wilcoxon_p": p_value,
                    "favorable_targets_source_lower": int((diffs < 0).sum()),
                    "targets": int(len(target)),
                    "paired_cells": int(group[f"{metric}_source"].notna().sum()),
                    "worst_assay_id_agreement_cells": int(group["worst_assay_id_agrees"].sum()),
                    "worst_assay_id_agreement_fraction": float(group["worst_assay_id_agrees"].mean()),
                }
            )

    pd.DataFrame(target_rows).to_csv(
        output_dir / "r2_9_quality_ablation_target_metrics.csv", index=False
    )
    summary = pd.DataFrame(summary_rows)
    summary["holm_p_across_10_quality_comparisons"] = holm_adjust(
        summary["wilcoxon_p"].to_numpy()
    )
    summary.to_csv(output_dir / "r2_9_quality_ablation_summary.csv", index=False)

    same_assay = pairs.loc[pairs["worst_assay_id_agrees"]].copy()
    same_assay.to_csv(
        output_dir / "r2_9_worst_assay_same_id_details.csv", index=False
    )
    mismatched = pairs.loc[~pairs["worst_assay_id_agrees"]].copy()
    mismatched.to_csv(
        output_dir / "r2_9_worst_assay_mismatched_id_cells.csv", index=False
    )

    manifest = {
        "pairing_unit": KEYS,
        "difference_direction": "source minus no-source; negative favors source context",
        "bootstrap": "20,000 target-level resamples with fixed seeds",
        "inputs": input_records,
        "rows": int(len(pairs)),
        "same_worst_assay_id_rows": int(pairs["worst_assay_id_agrees"].sum()),
        "different_worst_assay_id_rows": int((~pairs["worst_assay_id_agrees"]).sum()),
    }
    (output_dir / "r2_9_quality_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
