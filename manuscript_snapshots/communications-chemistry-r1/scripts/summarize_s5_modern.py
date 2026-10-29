#!/usr/bin/env python
"""Summarize the five-seed S5 graph-neural context without adding a new test family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["test_mae", "test_censored_nll", "test_censored_dir_acc", "worst_assay_mae"]
SEEDS = [2027, 2039, 2053, 2069, 2081]
BOOTSTRAP_SEED = 360_341
BOOTSTRAP_REPLICATES = 20_000


def read_runs(repo: Path, pattern: str) -> pd.DataFrame:
    paths = sorted(repo.glob(pattern))
    if len(paths) != len(SEEDS):
        raise RuntimeError(f"Expected five files for {pattern}, found {len(paths)}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame = frame.loc[frame["status"].eq("ok")].copy()
    if len(frame) != 100 or frame["target"].nunique() != 20 or sorted(frame["seed"].unique()) != SEEDS:
        raise RuntimeError(f"Incomplete S5 graph-neural runs for {pattern}")
    if frame.duplicated(["target", "split", "seed"]).any():
        raise RuntimeError(f"Duplicate target-split-seed rows for {pattern}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evidence_completion_2026-09-03/s5_modern"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    frames = {
        "D-MPNN bounded MSE": read_runs(
            repo,
            "results/modern/chemprop/chemprop_s5_bounded_mse_s*_s5_source5_20260903.csv",
        ),
        "AttentiveFP bounded MSE": read_runs(
            repo,
            "results/modern/attentivefp/attentivefp_s5_bounded_mse_s*_s5_source5_20260903.csv",
        ),
    }
    summary_rows = []
    for method, frame in frames.items():
        row: dict[str, object] = {
            "method": method,
            "runs": len(frame),
            "targets": frame["target"].nunique(),
            "seeds": frame["seed"].nunique(),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = frame[metric].mean()
            row[f"{metric}_std"] = frame[metric].std(ddof=1)
            row[f"{metric}_n"] = frame[metric].notna().sum()
        summary_rows.append(row)

    canal_runs = pd.read_csv(
        repo / "reports/evidence_completion_2026-09-03/source_context_s5_results_runs.csv"
    )
    canal = canal_runs.loc[
        canal_runs["treatment"].eq("censored_likelihood")
        & canal_runs["metadata_mode"].eq("source_light")
    ].copy()
    if len(canal) != 100:
        raise RuntimeError(f"Expected 100 CANAL S5 rows, found {len(canal)}")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    effect_rows = []
    for method, frame in frames.items():
        for metric in METRICS:
            left = canal.groupby("target")[metric].mean().rename("canal")
            right = frame.groupby("target")[metric].mean().rename("comparator")
            paired = pd.concat([left, right], axis=1).dropna()
            delta = (paired["canal"] - paired["comparator"]).to_numpy(dtype=float)
            indices = rng.integers(0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta)))
            means = delta[indices].mean(axis=1)
            low, high = np.quantile(means, [0.025, 0.975])
            effect_rows.append(
                {
                    "comparator": method,
                    "metric": metric,
                    "delta_canal_minus_comparator": delta.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "n_targets": len(delta),
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "inference_status": "exploratory; no p-value family added",
                }
            )

    pd.DataFrame(summary_rows).to_csv(output.with_name(output.name + "_summary.csv"), index=False)
    pd.DataFrame(effect_rows).to_csv(output.with_name(output.name + "_effects.csv"), index=False)
    pd.concat(frames.values(), ignore_index=True).to_csv(
        output.with_name(output.name + "_runs.csv"), index=False
    )
    manifest = {
        "seeds": SEEDS,
        "experimental_unit": "target after averaging five seeds",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "inference_status": "exploratory S5 neural context; no additional p-value family",
    }
    output.with_name(output.name + "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(output.with_name(output.name + "_summary.csv"))


if __name__ == "__main__":
    main()

