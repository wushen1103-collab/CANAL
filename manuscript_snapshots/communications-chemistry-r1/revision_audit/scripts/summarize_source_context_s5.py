#!/usr/bin/env python
"""Summarize the five-seed source-context-tail stress test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


BOOTSTRAP_SEED = 360_333
BOOTSTRAP_REPLICATES = 20_000
METRICS = {
    "test_mae": "lower",
    "test_censored_nll": "lower",
    "test_censored_dir_acc": "higher",
    "worst_assay_mae": "lower",
}


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(frame: pd.DataFrame) -> pd.Series:
    values = frame["p_raw"].to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return pd.Series(adjusted, index=frame.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evidence_completion_2026-09-03/source_context_s5_results"),
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path(
            "results/problem_check/"
            "problem_check_chembl35_ic50_top20_v2_s5_lgbm_s5_core_source_5seed_20260903.csv"
        ),
    )
    parser.add_argument(
        "--no-source-run",
        type=Path,
        default=Path(
            "results/problem_check/"
            "problem_check_chembl35_ic50_top20_v2_s5_lgbm_s5_core_no_source_5seed_20260903.csv"
        ),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    source_path = args.source_run if args.source_run.is_absolute() else repo / args.source_run
    no_source_path = (
        args.no_source_run if args.no_source_run.is_absolute() else repo / args.no_source_run
    )
    source = pd.read_csv(source_path)
    no_source = pd.read_csv(no_source_path)
    source = source.loc[source["status"].eq("ok")].copy()
    no_source = no_source.loc[no_source["status"].eq("ok")].copy()
    configs = {
        "Exact-only": source.loc[source["treatment"].eq("exact_only")].copy(),
        "Censored-as-exact": source.loc[source["treatment"].eq("censored_as_exact")].copy(),
        "CANAL": source.loc[source["treatment"].eq("censored_likelihood")].copy(),
        "CANAL without source metadata": no_source.copy(),
    }
    for label, frame in configs.items():
        if len(frame) != 100 or frame["target"].nunique() != 20 or frame["seed"].nunique() != 5:
            raise RuntimeError(f"{label}: expected 100 valid rows over 20 targets and five seeds")
        if frame.duplicated(["target", "split", "seed"]).any():
            raise RuntimeError(f"{label}: duplicate target-split-seed rows")

    summary_rows = []
    for label, frame in configs.items():
        row: dict[str, object] = {
            "configuration": label,
            "runs": len(frame),
            "targets": frame["target"].nunique(),
            "seeds": frame["seed"].nunique(),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = frame[metric].mean()
            row[f"{metric}_std"] = frame[metric].std(ddof=1)
            row[f"{metric}_n"] = frame[metric].notna().sum()
        summary_rows.append(row)

    full = configs["CANAL"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    test_rows = []
    for comparator_name in ["Exact-only", "Censored-as-exact", "CANAL without source metadata"]:
        comparator = configs[comparator_name]
        for metric, direction in METRICS.items():
            full_target = full.groupby("target")[metric].mean().rename("full")
            comparator_target = comparator.groupby("target")[metric].mean().rename("comparator")
            paired = pd.concat([full_target, comparator_target], axis=1).dropna()
            delta = (paired["full"] - paired["comparator"]).to_numpy(dtype=float)
            low, high = bootstrap_mean(delta, rng)
            try:
                p_raw = float(wilcoxon(delta).pvalue)
            except ValueError:
                p_raw = 1.0
            favorable = delta < 0 if direction == "lower" else delta > 0
            test_rows.append(
                {
                    "comparator": comparator_name,
                    "metric": metric,
                    "delta_canal_minus_comparator": delta.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "p_raw": p_raw,
                    "wins": int(favorable.sum()),
                    "n_targets": len(delta),
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
    tests = pd.DataFrame(test_rows)
    tests["p_holm"] = holm_adjust(tests)
    tests["holm_family_size"] = len(tests)
    pd.DataFrame(summary_rows).to_csv(output.with_name(output.name + "_summary.csv"), index=False)
    tests.to_csv(output.with_name(output.name + "_tests.csv"), index=False)
    pd.concat(configs.values(), ignore_index=True).to_csv(
        output.with_name(output.name + "_runs.csv"), index=False
    )
    manifest = {
        "source_run": str(source_path),
        "no_source_run": str(no_source_path),
        "experimental_unit": "target after averaging five seeds",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "holm_family_size": len(tests),
    }
    output.with_name(output.name + "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(output.with_name(output.name + "_summary.csv"))


if __name__ == "__main__":
    main()
