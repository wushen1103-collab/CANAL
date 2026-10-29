#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


BOOTSTRAP_SEED = 360037
BOOTSTRAP_REPLICATES = 20_000
METRICS = {
    "test_mae": False,
    "test_censored_nll": False,
    "test_censored_dir_acc": True,
    "worst_assay_mae": False,
}


def load_completed(directory: Path, pattern: str) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(path for path in directory.glob(pattern) if "partial" not in path.name)
    if len(files) != 4:
        raise RuntimeError(f"Expected four completed batch files for {pattern!r}, found {files}")
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True), files


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_adjust(p_values: pd.Series) -> pd.Series:
    order = np.argsort(p_values.to_numpy(dtype=float))
    ranked = p_values.to_numpy(dtype=float)[order]
    adjusted_ranked = np.maximum.accumulate((len(ranked) - np.arange(len(ranked))) * ranked)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    candidate, candidate_files = load_completed(
        args.result_dir, "problem_check_chembl36_ic50_release_update_lgbm_commchem_valonly_batch*_locked5.csv"
    )
    locked, locked_files = load_completed(
        args.result_dir, "problem_check_chembl36_ic50_release_update_lgbm_chembl36_update_batch*_locked5.csv"
    )
    candidate = candidate.loc[candidate["treatment"].eq("censored_likelihood")].copy()
    candidate["treatment"] = "canal_validation_ranked"
    locked = locked.loc[locked["treatment"].isin(["censored_as_exact", "censored_likelihood"])].copy()
    locked["treatment"] = locked["treatment"].replace({"censored_likelihood": "canal_final"})
    frame = pd.concat([locked, candidate], ignore_index=True)

    expected = {"censored_as_exact", "canal_final", "canal_validation_ranked"}
    if set(frame["treatment"].unique()) != expected:
        raise RuntimeError(f"Unexpected treatments: {sorted(frame['treatment'].unique())}")
    counts = frame.groupby("treatment").size()
    if not counts.eq(100).all():
        raise RuntimeError(f"Expected 100 rows per treatment, found {counts.to_dict()}")
    coverage = frame.groupby("treatment").agg(targets=("target", "nunique"), seeds=("seed", "nunique"))
    if not (coverage["targets"].eq(20).all() and coverage["seeds"].eq(5).all()):
        raise RuntimeError(f"Incomplete coverage: {coverage.to_dict('index')}")

    summary_rows = []
    for treatment, block in frame.groupby("treatment"):
        for metric in METRICS:
            values = block[metric].dropna()
            summary_rows.append(
                {
                    "treatment": treatment,
                    "metric": metric,
                    "target_seed_cells": int(len(values)),
                    "mean": float(values.mean()),
                    "sd": float(values.std()),
                }
            )
    summary = pd.DataFrame(summary_rows)

    target = frame.groupby(["target", "treatment"])[list(METRICS)].mean().reset_index()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    inference_rows = []
    for comparator in ["censored_as_exact", "canal_final"]:
        for metric, higher_is_better in METRICS.items():
            wide = target.pivot(index="target", columns="treatment", values=metric).dropna(
                subset=[comparator, "canal_validation_ranked"]
            )
            delta = (wide["canal_validation_ranked"] - wide[comparator]).to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_ci(delta, rng)
            p_value = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
            wins = int((delta > 0).sum() if higher_is_better else (delta < 0).sum())
            inference_rows.append(
                {
                    "candidate": "canal_validation_ranked",
                    "comparator": comparator,
                    "metric": metric,
                    "experimental_unit": "target",
                    "n_targets": int(len(delta)),
                    "delta_candidate_minus_comparator": float(delta.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "wilcoxon_p": p_value,
                    "target_wins": wins,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
    inference = pd.DataFrame(inference_rows)
    inference["holm_p"] = inference.groupby("comparator", group_keys=False)["wilcoxon_p"].apply(holm_adjust)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "validation_only_transfer_cells.csv", index=False)
    target.to_csv(args.out_dir / "validation_only_transfer_target_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "validation_only_transfer_summary.csv", index=False)
    inference.to_csv(args.out_dir / "validation_only_transfer_inference.csv", index=False)
    manifest = {
        "candidate_files": [str(path) for path in candidate_files],
        "locked_comparator_files": [str(path) for path in locked_files],
        "candidate_hyperparameters": {"censor_weight": 0.5, "censored_anchor_weight": 0.5},
        "selection_rule": "first-ranked candidate in the retrospective ChEMBL35 validation audit",
        "external_test": "untouched ChEMBL36 release-update cohort",
        "targets": 20,
        "seeds": [2027, 2039, 2053, 2069, 2081],
        "experimental_unit": "target",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    (args.out_dir / "validation_only_transfer_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(inference.to_string(index=False))


if __name__ == "__main__":
    main()
