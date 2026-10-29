#!/usr/bin/env python3
"""Add target-level uncertainty and multiplicity control to Reviewer 2 metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


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


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    audit_dir = args.audit_dir.resolve()
    target = pd.read_csv(audit_dir / "r2_8_r2_10d_target_metrics.csv")
    matched = target.loc[target["treatment"].eq("censored_as_exact")].set_index("target_chembl_id")
    canal = target.loc[target["treatment"].eq("censored_likelihood")].set_index("target_chembl_id")

    rng = np.random.default_rng(20261028)
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        difference = (canal[metric] - matched[metric]).dropna().to_numpy(dtype=float)
        draws = rng.choice(difference, size=(20000, len(difference)), replace=True).mean(axis=1)
        try:
            p_value = float(wilcoxon(difference, alternative="two-sided").pvalue)
        except ValueError:
            p_value = 1.0
        rows.append(
            {
                "metric": metric,
                "difference": "CANAL minus censored-as-exact",
                "target_mean_difference": float(difference.mean()),
                "bootstrap_ci_low": float(np.quantile(draws, 0.025)),
                "bootstrap_ci_high": float(np.quantile(draws, 0.975)),
                "wilcoxon_p": p_value,
                "targets": int(len(difference)),
                "bootstrap_replicates": 20000,
                "bootstrap_seed": 20261028,
            }
        )
    output = pd.DataFrame(rows)
    output["holm_p"] = holm_adjust(output["wilcoxon_p"].to_numpy())
    output.to_csv(audit_dir / "r2_8_r2_10d_paired_target_inference.csv", index=False)
    s3 = pd.read_csv(audit_dir / "r2_11_s3_cell_metrics.csv")
    s3_target = s3.groupby(["treatment", "method", "target_chembl_id"], as_index=False)[METRICS].mean()
    s3_summary = s3_target.groupby(["treatment", "method"], as_index=False)[METRICS].mean()
    s3_summary["targets"] = s3_target["target_chembl_id"].nunique()
    s3_summary["cells"] = s3.groupby("treatment").size().reindex(s3_summary["treatment"]).to_numpy()
    s3_target.to_csv(audit_dir / "r2_11_s3_primary_target_metrics.csv", index=False)
    s3_summary.to_csv(audit_dir / "r2_11_s3_primary_target_balanced_summary.csv", index=False)
    print(output.to_string(index=False))
    print(s3_summary.to_string(index=False))


if __name__ == "__main__":
    main()
