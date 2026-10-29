#!/usr/bin/env python3
"""Create a common-cell comparison for the published UQ4DD MVE/Tobit baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["target_chembl_id", "split", "seed"]
METRICS = [
    "exact_mae",
    "exact_rmse",
    "exact_r2",
    "exact_spearman",
    "exact_kendall",
    "censored_nll_mean",
    "censored_nll_median",
    "cdir",
    "violation_distance",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/reviewer2_validation_2026-10-28/uq4dd_baseline"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    uq_paths = sorted((root / "results" / "reviewer2").glob("uq4dd_mve_tobit_shard[0-3].csv"))
    if len(uq_paths) != 4:
        raise FileNotFoundError(f"Expected four UQ4DD shards, found {len(uq_paths)}")
    uq = pd.concat([pd.read_csv(path) for path in uq_paths], ignore_index=True)
    if not uq["status"].eq("ok").all() or uq.duplicated(["target", "split", "seed"]).any():
        raise ValueError("UQ4DD shards contain failed or duplicate cells")
    uq = uq.rename(
        columns={
            "target": "target_chembl_id",
            "test_mae": "exact_mae",
            "test_rmse": "exact_rmse",
            "test_r2": "exact_r2",
            "test_spearman": "exact_spearman",
            "test_kendall": "exact_kendall",
            "test_censored_nll": "censored_nll_mean",
            "test_censored_nll_median": "censored_nll_median",
            "test_censored_dir_acc": "cdir",
            "test_violation_distance": "violation_distance",
        }
    )
    uq["method"] = "UQ4DD-MVE/Tobit (published protocol adaptation)"

    cell_path = (
        root
        / "reports"
        / "reviewer2_validation_2026-10-28"
        / "final_300tree"
        / "r2_8_r2_10d_r2_11_test_cell_metrics.csv"
    )
    existing = pd.read_csv(cell_path)
    common_keys = uq[KEYS].drop_duplicates()
    existing_common = existing.merge(common_keys, on=KEYS, how="inner", validate="many_to_one")
    comparison = pd.concat(
        [existing_common[["method"] + KEYS + METRICS], uq[["method"] + KEYS + METRICS]],
        ignore_index=True,
    )
    comparison.to_csv(output_dir / "r2_10b_uq4dd_common_cell_metrics.csv", index=False)

    coverage = (
        comparison.groupby("method", as_index=False)
        .agg(cells=("seed", "size"), targets=("target_chembl_id", "nunique"))
        .sort_values("method")
    )
    coverage.to_csv(output_dir / "r2_10b_uq4dd_common_cell_coverage.csv", index=False)

    expected = pd.MultiIndex.from_product(
        [sorted(uq["target_chembl_id"].unique()), ["S1", "S2", "S3", "S4"], [2027, 2039, 2053, 2069, 2081]],
        names=KEYS,
    ).to_frame(index=False)
    missing = expected.merge(uq[KEYS], on=KEYS, how="left", indicator=True)
    missing = missing.loc[missing["_merge"].eq("left_only")].drop(columns="_merge")
    missing.to_csv(output_dir / "r2_10b_uq4dd_unavailable_cells.csv", index=False)

    target_metrics = comparison.groupby(["method", "target_chembl_id"], as_index=False)[METRICS].mean()
    target_metrics.to_csv(output_dir / "r2_10b_uq4dd_target_metrics.csv", index=False)
    summary = target_metrics.groupby("method", as_index=False)[METRICS].mean()
    summary["targets"] = 20
    summary["common_cells"] = len(uq)
    summary.to_csv(output_dir / "r2_10b_uq4dd_target_balanced_summary.csv", index=False)

    canal = target_metrics.loc[target_metrics["method"].eq("CANAL")].set_index("target_chembl_id")
    uq_target = target_metrics.loc[target_metrics["method"].str.startswith("UQ4DD")].set_index("target_chembl_id")
    paired_rows: list[dict[str, object]] = []
    higher_better = {"exact_r2", "exact_spearman", "exact_kendall", "cdir"}
    for metric_index, metric in enumerate(METRICS):
        raw_difference = canal[metric] - uq_target[metric]
        favorable_difference = raw_difference if metric in higher_better else -raw_difference
        ci_low, ci_high = bootstrap_ci(raw_difference.to_numpy(), 20261028 + metric_index)
        paired_rows.append(
            {
                "metric": metric,
                "canal": float(canal[metric].mean()),
                "uq4dd": float(uq_target[metric].mean()),
                "canal_minus_uq4dd": float(raw_difference.mean()),
                "bootstrap_ci_low": float(ci_low),
                "bootstrap_ci_high": float(ci_high),
                "targets_favoring_canal": int((favorable_difference > 0).sum()),
                "targets": int(len(raw_difference)),
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_dir / "r2_10b_uq4dd_canal_paired_effects.csv", index=False)

    manifest = {
        "published_method": "Svensson et al., Artificial Intelligence in the Life Sciences 7 (2025) 100128",
        "repository": "https://github.com/MolecularAI/uq4dd",
        "repository_commit": str(uq["uq4dd_commit"].iloc[0]),
        "adaptation": "Published MVE architecture and Gaussian Tobit objective evaluated on CANAL's fixed IC50 splits; no source metadata.",
        "aggregation": "Mean over common split-by-seed cells within each target, followed by equal-weight mean across 20 targets.",
        "available_cells": int(len(uq)),
        "unavailable_cells": int(len(missing)),
        "unavailable_reason": "run_one requires at least 20 validation records; unavailable cells are listed explicitly.",
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in uq_paths]
        + [{"path": str(cell_path), "sha256": sha256(cell_path)}],
    }
    (output_dir / "r2_10b_uq4dd_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(coverage.to_string(index=False))
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
    print(missing.to_string(index=False))


if __name__ == "__main__":
    main()
