#!/usr/bin/env python
"""Evaluate Gaussian predictive calibration on exact-label test records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon


LEVELS = [0.50, 0.70, 0.80, 0.90, 0.95]
BOOTSTRAP_SEED = 360_133
BOOTSTRAP_REPLICATES = 20_000
METHODS = {
    "exact_only": "Exact-only",
    "censored_as_exact": "Censored-as-exact",
    "censored_likelihood": "CANAL",
}


def normal_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.maximum(sigma.astype(float), 1e-6)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    return (
        upper
        - lower
        + (2.0 / alpha) * (lower - y) * (y < lower)
        + (2.0 / alpha) * (y - upper) * (y > upper)
    )


def bootstrap_target_mean(values: pd.Series, rng: np.random.Generator) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    draws = rng.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))
    means = array[draws].mean(axis=1)
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
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evidence_completion_2026-09-03/calibration"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_parquet(path) for path in args.predictions]
    predictions = pd.concat(frames, ignore_index=True)
    if predictions["seed"].nunique() != 5:
        raise RuntimeError(f"Expected five seeds, observed {predictions['seed'].nunique()}")
    missing = set(METHODS) - set(predictions["treatment"].unique())
    if missing:
        raise RuntimeError(f"Missing treatments: {sorted(missing)}")
    exact = predictions.loc[predictions["is_exact"].fillna(False).astype(bool)].copy()
    exact["method_label"] = exact["treatment"].map(METHODS)
    y = exact["p_activity"].to_numpy(dtype=float)
    mu = exact["pred"].to_numpy(dtype=float)
    sigma = exact["sigma"].to_numpy(dtype=float)
    exact["crps"] = normal_crps(y, mu, sigma)
    exact["abs_error_calibration"] = np.abs(y - mu)
    for level in LEVELS:
        alpha = 1.0 - level
        z = norm.ppf(1.0 - alpha / 2.0)
        lower = mu - z * sigma
        upper = mu + z * sigma
        suffix = str(int(level * 100))
        exact[f"covered_{suffix}"] = (y >= lower) & (y <= upper)
        exact[f"width_{suffix}"] = upper - lower
        exact[f"interval_score_{suffix}"] = interval_score(y, lower, upper, alpha)

    group_columns = ["treatment", "method_label", "target_chembl_id", "split", "seed"]
    aggregate_spec = {
        "rows": ("row_id", "size"),
        "crps": ("crps", "mean"),
        "gaussian_nll": ("gaussian_nll", "mean"),
        "mae": ("abs_error_calibration", "mean"),
    }
    for level in LEVELS:
        suffix = str(int(level * 100))
        aggregate_spec[f"coverage_{suffix}"] = (f"covered_{suffix}", "mean")
        aggregate_spec[f"width_{suffix}"] = (f"width_{suffix}", "mean")
        aggregate_spec[f"interval_score_{suffix}"] = (f"interval_score_{suffix}", "mean")
    cells = exact.groupby(group_columns, sort=True).agg(**aggregate_spec).reset_index()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    score_rows = []
    calibration_rows = []
    for treatment, method_cells in cells.groupby("treatment", sort=False):
        for split_name, subset in [("Overall", method_cells), *list(method_cells.groupby("split", sort=True))]:
            if isinstance(subset, tuple):
                split_name, subset = subset
            target_scores = subset.groupby("target_chembl_id")[["crps", "gaussian_nll", "mae"]].mean()
            for metric in ["crps", "gaussian_nll", "mae"]:
                low, high = bootstrap_target_mean(target_scores[metric], rng)
                score_rows.append(
                    {
                        "treatment": treatment,
                        "method": METHODS[treatment],
                        "split": split_name,
                        "metric": metric,
                        "target_balanced_mean": target_scores[metric].mean(),
                        "target_std": target_scores[metric].std(ddof=1),
                        "ci_low": low,
                        "ci_high": high,
                        "n_targets": len(target_scores),
                        "rows": int(subset["rows"].sum()),
                    }
                )
            for level in LEVELS:
                suffix = str(int(level * 100))
                target_level = subset.groupby("target_chembl_id")[[
                    f"coverage_{suffix}", f"width_{suffix}", f"interval_score_{suffix}"
                ]].mean()
                coverage_low, coverage_high = bootstrap_target_mean(target_level[f"coverage_{suffix}"], rng)
                calibration_rows.append(
                    {
                        "treatment": treatment,
                        "method": METHODS[treatment],
                        "split": split_name,
                        "nominal_coverage": level,
                        "empirical_coverage": target_level[f"coverage_{suffix}"].mean(),
                        "absolute_calibration_error": abs(target_level[f"coverage_{suffix}"].mean() - level),
                        "mean_interval_width": target_level[f"width_{suffix}"].mean(),
                        "mean_interval_score": target_level[f"interval_score_{suffix}"].mean(),
                        "coverage_ci_low": coverage_low,
                        "coverage_ci_high": coverage_high,
                        "n_targets": len(target_level),
                        "rows": int(subset["rows"].sum()),
                    }
                )

    paired_rows = []
    overall = cells.groupby(["treatment", "target_chembl_id"], sort=True).mean(numeric_only=True)
    for comparator in ["exact_only", "censored_as_exact"]:
        for metric in ["crps", "gaussian_nll", "interval_score_90"]:
            full = overall.loc["censored_likelihood", metric]
            base = overall.loc[comparator, metric]
            paired = pd.concat([full.rename("full"), base.rename("base")], axis=1).dropna()
            delta = (paired["full"] - paired["base"]).to_numpy(dtype=float)
            low, high = bootstrap_target_mean(pd.Series(delta), rng)
            try:
                p_raw = float(wilcoxon(delta).pvalue)
            except ValueError:
                p_raw = 1.0
            paired_rows.append(
                {
                    "comparator": METHODS[comparator],
                    "metric": metric,
                    "delta_canal_minus_comparator": delta.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "p_raw": p_raw,
                    "wins": int((delta < 0).sum()),
                    "n_targets": len(delta),
                }
            )
    paired_tests = pd.DataFrame(paired_rows)
    paired_tests["p_holm"] = holm_adjust(paired_tests)
    paired_tests["holm_family_size"] = len(paired_tests)

    pd.DataFrame(score_rows).to_csv(output.with_name(output.name + "_proper_scores.csv"), index=False)
    pd.DataFrame(calibration_rows).to_csv(output.with_name(output.name + "_coverage.csv"), index=False)
    paired_tests.to_csv(output.with_name(output.name + "_paired_tests.csv"), index=False)
    cells.to_csv(output.with_name(output.name + "_target_split_seed.csv"), index=False)
    manifest = {
        "predictions": [str(path.resolve()) for path in args.predictions],
        "levels": LEVELS,
        "experimental_unit": "target after averaging split and seed",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "paired_holm_family": len(paired_tests),
    }
    output.with_name(output.name + "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(output.with_name(output.name + "_coverage.csv"))


if __name__ == "__main__":
    main()
