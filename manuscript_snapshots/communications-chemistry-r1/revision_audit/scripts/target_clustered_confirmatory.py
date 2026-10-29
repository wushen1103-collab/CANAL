#!/usr/bin/env python
"""Reproduce the target-clustered confirmatory tests reported in the manuscript."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


BOOTSTRAP_SEED = 330033
BOOTSTRAP_REPLICATES = 20_000

MAIN_QUERY = {
    "dataset": "chembl35_ic50_top20_v2",
    "model": "lgbm",
    "feature_set": "ecfp_desc",
    "metadata_mode": "source_light",
    "treatment": "censored_likelihood",
    "censor_weight": 0.25,
    "censored_anchor_weight": 1.0,
    "right_censor_weight": 0.1,
    "right_censored_anchor_weight": 0.0,
}

BASELINES = [
    (
        "Same-backbone censored-as-exact",
        {
            "dataset": "chembl35_ic50_top20_v2",
            "model": "lgbm",
            "feature_set": "ecfp_desc",
            "metadata_mode": "source_light",
            "treatment": "censored_as_exact",
        },
    ),
    (
        "XGB exact-only",
        {
            "dataset": "chembl35_ic50_top20_v2",
            "model": "xgb",
            "feature_set": "ecfp_desc",
            "metadata_mode": "source_light",
            "treatment": "exact_only",
        },
    ),
    (
        "Pure censored likelihood",
        {
            "dataset": "chembl35_ic50_top20_v2",
            "model": "lgbm",
            "feature_set": "ecfp_desc",
            "metadata_mode": "source_light",
            "treatment": "censored_likelihood",
            "censor_weight": 1.0,
            "censored_anchor_weight": 0.0,
            "right_censor_weight": None,
            "right_censored_anchor_weight": None,
        },
    ),
    (
        "Standard linear Tobit",
        {
            "dataset": "chembl35_ic50_top20_v2",
            "model": "tobit",
            "feature_set": "desc",
            "metadata_mode": "source_light",
            "treatment": "tobit_standard",
        },
    ),
]

METRICS = ["test_mae", "test_censored_nll", "test_censored_dir_acc"]


def holm_adjust(rows: list[dict[str, object]]) -> None:
    """Attach monotone Holm-adjusted p-values over the complete test family."""
    order = sorted(range(len(rows)), key=lambda i: float(rows[i]["p_raw"]))
    running_max = 0.0
    for rank, row_index in enumerate(order):
        candidate = min(1.0, (len(rows) - rank) * float(rows[row_index]["p_raw"]))
        running_max = max(running_max, candidate)
        rows[row_index]["p_holm"] = running_max


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/submission_completion_2026-08-12.target_clustered.csv"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))

    from scripts.summarize_submission_completion import read_problem_files, select_row

    frame = read_problem_files(repo / "results" / "problem_check")
    candidate = select_row(frame, **MAIN_QUERY)
    if len(candidate) != 400 or candidate["target"].nunique() != 20:
        raise RuntimeError("Expected 400 CANAL runs over 20 targets")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for baseline_name, baseline_query in BASELINES:
        baseline = select_row(frame, **baseline_query)
        if len(baseline) != 400 or baseline["target"].nunique() != 20:
            raise RuntimeError(f"Expected 400 runs over 20 targets for {baseline_name}")
        for metric in METRICS:
            main_target = candidate.groupby("target")[metric].mean().rename("main")
            base_target = baseline.groupby("target")[metric].mean().rename("baseline")
            paired = pd.concat([main_target, base_target], axis=1).dropna()
            delta = (paired["main"] - paired["baseline"]).to_numpy(dtype=float)
            indices = rng.integers(
                0,
                len(delta),
                size=(BOOTSTRAP_REPLICATES, len(delta)),
            )
            bootstrap_mean = delta[indices].mean(axis=1)
            lower_is_better = metric != "test_censored_dir_acc"
            rows.append(
                {
                    "comparator": baseline_name,
                    "metric": metric,
                    "experimental_unit": "target",
                    "n_targets": len(delta),
                    "delta_main_minus_baseline": float(delta.mean()),
                    "ci_low": float(np.quantile(bootstrap_mean, 0.025)),
                    "ci_high": float(np.quantile(bootstrap_mean, 0.975)),
                    "p_raw": float(wilcoxon(delta).pvalue),
                    "wins": int((delta < 0).sum() if lower_is_better else (delta > 0).sum()),
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "holm_family_size": len(BASELINES) * len(METRICS),
                }
            )

    holm_adjust(rows)
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
