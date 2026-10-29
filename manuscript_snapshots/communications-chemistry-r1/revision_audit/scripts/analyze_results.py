#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def infer_dataset_from_result_name(path: Path) -> str:
    stem = path.stem
    prefix = "problem_check_"
    if not stem.startswith(prefix):
        return "unknown"
    remainder = stem[len(prefix) :]
    for marker in ("_xgb", "_rf", "_lgbm"):
        if marker in remainder:
            return remainder.split(marker, 1)[0]
    return "unknown"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"].eq("ok")].copy()
    metrics = [
        "test_mae",
        "test_rmse",
        "test_spearman",
        "test_nll",
        "test_coverage90",
        "test_censored_nll",
        "test_censored_dir_acc",
        "worst_assay_mae",
    ]
    rows = []
    group_cols = ["endpoint", "split", "treatment", "model"]
    if "dataset" in ok.columns:
        ok["dataset"] = ok["dataset"].fillna("unknown")
        group_cols.insert(0, "dataset")
    if "feature_set" in ok.columns:
        ok["feature_set"] = ok["feature_set"].fillna("ecfp")
        group_cols.append("feature_set")
    if "censor_weight" in ok.columns:
        ok["censor_weight"] = ok["censor_weight"].fillna(-1.0)
        group_cols.append("censor_weight")
    if "censored_anchor_weight" in ok.columns:
        ok["censored_anchor_weight"] = ok["censored_anchor_weight"].fillna(-1.0)
        group_cols.append("censored_anchor_weight")
    if "right_censor_weight" in ok.columns:
        ok["right_censor_weight"] = ok["right_censor_weight"].fillna(-1.0)
        group_cols.append("right_censor_weight")
    if "right_censored_anchor_weight" in ok.columns:
        ok["right_censored_anchor_weight"] = ok["right_censored_anchor_weight"].fillna(-1.0)
        group_cols.append("right_censored_anchor_weight")
    if "metadata_mode" in ok.columns:
        ok["metadata_mode"] = ok["metadata_mode"].fillna("none")
        group_cols.append("metadata_mode")
    if "sigma_calibration" in ok.columns:
        ok["sigma_calibration"] = ok["sigma_calibration"].fillna("exact")
        group_cols.append("sigma_calibration")
    if "posthoc_bias" in ok.columns:
        ok["posthoc_bias"] = ok["posthoc_bias"].fillna("none")
        group_cols.append("posthoc_bias")
    if "posthoc_lambda" in ok.columns:
        ok["posthoc_lambda"] = ok["posthoc_lambda"].fillna(-1.0)
        group_cols.append("posthoc_lambda")
    for keys, group in ok.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["runs"] = len(group)
        for metric in metrics:
            if metric in group:
                vals = pd.to_numeric(group[metric], errors="coerce")
                row[f"{metric}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
                row[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def add_delta(summary: pd.DataFrame, baseline: str = "exact_only") -> pd.DataFrame:
    out = summary.copy()
    key_cols = ["endpoint", "split", "model"]
    if "dataset" in out.columns:
        key_cols.insert(0, "dataset")
    if "metadata_mode" in out.columns:
        key_cols.append("metadata_mode")
    if "sigma_calibration" in out.columns:
        key_cols.append("sigma_calibration")
    if "posthoc_bias" in out.columns:
        key_cols.append("posthoc_bias")
    if "posthoc_lambda" in out.columns:
        key_cols.append("posthoc_lambda")
    for metric in ["test_mae_mean", "test_nll_mean", "test_censored_nll_mean", "worst_assay_mae_mean"]:
        if metric not in out:
            continue
        base = out[out["treatment"].eq(baseline)][key_cols + [metric]].rename(columns={metric: f"{metric}_baseline"})
        out = out.merge(base, on=key_cols, how="left")
        out[f"{metric}_delta_vs_{baseline}"] = out[metric] - out[f"{metric}_baseline"]
        out = out.drop(columns=[f"{metric}_baseline"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/problem_check"))
    parser.add_argument("--out", type=Path, default=Path("reports/problem_check_summary.csv"))
    args = parser.parse_args()

    files = sorted(args.results_dir.glob("problem_check_*.csv"))
    files = [f for f in files if "partial" not in f.name]
    if not files:
        raise FileNotFoundError(f"No result CSVs found in {args.results_dir}")
    frames = []
    for f in files:
        frame = pd.read_csv(f)
        if "dataset" not in frame.columns:
            frame["dataset"] = infer_dataset_from_result_name(f)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    summary = add_delta(summarize(df))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
