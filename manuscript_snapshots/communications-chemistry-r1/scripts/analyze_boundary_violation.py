#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


BOOTSTRAP_SEED = 360035
BOOTSTRAP_REPLICATES = 20_000
CANAL = "censored_likelihood"
MATCHED = "censored_as_exact"


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.prediction_dir.glob("ic50_lgbm_all_treatments_s*_20260903.parquet"))
    if len(files) != 5:
        raise RuntimeError(f"Expected five prediction files, found {len(files)}")
    columns = [
        "target_chembl_id", "split", "seed", "treatment", "p_relation",
        "p_activity", "pred",
    ]
    frame = pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)
    frame = frame.loc[
        frame["treatment"].isin([MATCHED, CANAL]) & frame["p_relation"].ne("=")
    ].copy()
    frame["direction"] = np.where(frame["p_relation"].eq("<"), "left", "right")
    frame["violation_distance"] = np.where(
        frame["p_relation"].eq("<"),
        np.maximum(frame["pred"] - frame["p_activity"], 0.0),
        np.maximum(frame["p_activity"] - frame["pred"], 0.0),
    )
    frame["violates"] = frame["violation_distance"].gt(0)

    cell = (
        frame.groupby(["target_chembl_id", "split", "seed", "treatment", "direction"])
        .agg(
            n=("violation_distance", "size"),
            mean_violation=("violation_distance", "mean"),
            violation_rate=("violates", "mean"),
            mean_distance_when_violating=(
                "violation_distance",
                lambda x: float(x[x.gt(0)].mean()) if x.gt(0).any() else 0.0,
            ),
        )
        .reset_index()
    )
    pooled = (
        frame.groupby(["target_chembl_id", "split", "seed", "treatment"])
        .agg(
            n=("violation_distance", "size"),
            mean_violation=("violation_distance", "mean"),
            violation_rate=("violates", "mean"),
            mean_distance_when_violating=(
                "violation_distance",
                lambda x: float(x[x.gt(0)].mean()) if x.gt(0).any() else 0.0,
            ),
        )
        .reset_index()
        .assign(direction="all")
    )
    cell = pd.concat([cell, pooled], ignore_index=True)

    summary = (
        cell.groupby(["direction", "treatment"])
        .agg(
            cells=("mean_violation", "size"),
            mean_violation=("mean_violation", "mean"),
            sd_violation=("mean_violation", "std"),
            mean_violation_rate=("violation_rate", "mean"),
            sd_violation_rate=("violation_rate", "std"),
            mean_distance_when_violating=("mean_distance_when_violating", "mean"),
            sd_distance_when_violating=("mean_distance_when_violating", "std"),
        )
        .reset_index()
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    inference_rows = []
    for direction in ["all", "left", "right"]:
        block = cell.loc[cell["direction"].eq(direction)]
        target = (
            block.groupby(["target_chembl_id", "treatment"])["mean_violation"]
            .mean()
            .unstack()
            .dropna(subset=[MATCHED, CANAL])
        )
        delta = (target[CANAL] - target[MATCHED]).to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(delta, rng)
        p_value = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
        inference_rows.append(
            {
                "direction": direction,
                "experimental_unit": "target",
                "n_targets": int(len(delta)),
                "delta_canal_minus_matched": float(delta.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "wilcoxon_p": p_value,
                "target_wins": int((delta < 0).sum()),
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cell.to_csv(args.out_dir / "boundary_violation_cells.csv", index=False)
    summary.to_csv(args.out_dir / "boundary_violation_summary.csv", index=False)
    inference = pd.DataFrame(inference_rows)
    inference.to_csv(args.out_dir / "boundary_violation_target_inference.csv", index=False)
    manifest = {
        "inputs": [str(path) for path in files],
        "rows": int(len(frame)),
        "definition": (
            "Distance beyond the incompatible side of a reported pActivity bound, with "
            "zero assigned to bound-compatible predictions."
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    (args.out_dir / "boundary_violation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(inference.to_string(index=False))


if __name__ == "__main__":
    main()
