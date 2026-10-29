#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


SEEDS_5 = [2027, 2039, 2053, 2069, 2081]


def read_problem_files(results_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(results_dir.glob("problem_check_chembl35_*_top20_v2_*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "status" not in df or "seed" not in df:
            continue
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["status"].eq("ok")].copy()
    df = df[df["seed"].isin(SEEDS_5)]
    for col, default in [
        ("dataset", "unknown"),
        ("model", "unknown"),
        ("feature_set", "ecfp"),
        ("metadata_mode", "none"),
        ("sigma_calibration", "exact"),
        ("posthoc_bias", "none"),
    ]:
        if col not in df:
            df[col] = default
        df[col] = df[col].fillna(default)
    for col in [
        "censor_weight",
        "censored_anchor_weight",
        "right_censor_weight",
        "right_censored_anchor_weight",
        "posthoc_lambda",
    ]:
        if col not in df:
            df[col] = np.nan
    key_cols = [
        "dataset",
        "endpoint",
        "target",
        "split",
        "seed",
        "treatment",
        "model",
        "feature_set",
        "metadata_mode",
        "sigma_calibration",
        "censor_weight",
        "censored_anchor_weight",
        "right_censor_weight",
        "right_censored_anchor_weight",
        "posthoc_bias",
        "posthoc_lambda",
    ]
    df = df.sort_values("source_file").drop_duplicates(key_cols, keep="last")
    return df


def fmt_config(value: object) -> str:
    try:
        num = float(value)
    except Exception:
        return ""
    if not np.isfinite(num):
        return ""
    return f"{num:g}"


def method_label(row: pd.Series) -> str:
    parts = [
        str(row["model"]).upper(),
        str(row["feature_set"]),
        str(row["metadata_mode"]),
        str(row["treatment"]),
    ]
    if row["treatment"] == "censored_likelihood":
        cfg = []
        for col, label in [
            ("censor_weight", "cw"),
            ("censored_anchor_weight", "aw"),
            ("right_censor_weight", "right_cw"),
            ("right_censored_anchor_weight", "right_aw"),
        ]:
            val = fmt_config(row.get(col, np.nan))
            if val:
                cfg.append(f"{label}={val}")
        if cfg:
            parts.append("(" + ", ".join(cfg) + ")")
    return " ".join(p for p in parts if p)


def summarize_groups(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "endpoint",
        "treatment",
        "model",
        "feature_set",
        "metadata_mode",
        "sigma_calibration",
        "censor_weight",
        "censored_anchor_weight",
        "right_censor_weight",
        "right_censored_anchor_weight",
        "posthoc_bias",
        "posthoc_lambda",
    ]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["method"] = method_label(pd.Series(row))
        row["runs"] = len(group)
        row["targets"] = int(group["target"].nunique())
        row["splits"] = int(group["split"].nunique())
        row["seeds"] = int(group["seed"].nunique())
        for metric in ["test_mae", "test_censored_nll", "test_censored_dir_acc", "worst_assay_mae"]:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise(df: pd.DataFrame, main_query: dict[str, object], baselines: list[tuple[str, dict[str, object]]]) -> pd.DataFrame:
    def select(query: dict[str, object]) -> pd.DataFrame:
        out = df.copy()
        for col, val in query.items():
            if val is None:
                out = out[out[col].isna()]
            else:
                out = out[out[col].eq(val)]
        return out

    main = select(main_query)
    key = ["target", "split", "seed"]
    rows = []
    for name, query in baselines:
        base = select(query)
        merged = main.merge(base, on=key, suffixes=("_main", "_base"))
        for metric in ["test_mae", "test_censored_nll", "test_censored_dir_acc", "worst_assay_mae"]:
            m = pd.to_numeric(merged[f"{metric}_main"], errors="coerce")
            b = pd.to_numeric(merged[f"{metric}_base"], errors="coerce")
            mask = m.notna() & b.notna()
            diff = m[mask] - b[mask]
            if len(diff) == 0:
                continue
            try:
                p = float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0
            except Exception:
                p = np.nan
            lower_is_better = metric != "test_censored_dir_acc"
            wins = int((diff < 0).sum()) if lower_is_better else int((diff > 0).sum())
            rows.append(
                {
                    "baseline": name,
                    "metric": metric,
                    "n": int(len(diff)),
                    "delta_main_minus_baseline": float(diff.mean()),
                    "p_wilcoxon": p,
                    "wins": wins,
                }
            )
    return pd.DataFrame(rows)


def fmt(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def write_md(rank: pd.DataFrame, endpoints: pd.DataFrame, pairs: pd.DataFrame, out: Path) -> None:
    lines = ["# Five-Seed Completion Summary", "", "Seeds: 2027/2039/2053/2069/2081.", ""]
    lines.append("## IC50 Fine-Grained Rank")
    show = rank.copy()
    show = show[["method", "runs", "targets", "splits", "seeds", "test_mae_mean", "test_censored_nll_mean", "test_censored_dir_acc_mean", "worst_assay_mae_mean"]]
    show.columns = ["method", "runs", "targets", "splits", "seeds", "MAE", "censored NLL", "censored dir", "worst assay MAE"]
    lines.append(show.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    if not pairs.empty:
        lines.append("## Key Pairwise Checks")
        lines.append(pairs.to_markdown(index=False, floatfmt=".6g"))
        lines.append("")
    if not endpoints.empty:
        lines.append("## Endpoint Support")
        ep = endpoints.copy()
        ep["MAE"] = [fmt(m, s) for m, s in zip(ep["test_mae_mean"], ep["test_mae_std"])]
        ep["censored NLL"] = [fmt(m, s) for m, s in zip(ep["test_censored_nll_mean"], ep["test_censored_nll_std"])]
        ep["censored dir"] = [fmt(m, s) for m, s in zip(ep["test_censored_dir_acc_mean"], ep["test_censored_dir_acc_std"])]
        lines.append(ep[["dataset", "endpoint", "method", "runs", "seeds", "MAE", "censored NLL", "censored dir"]].to_markdown(index=False))
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/problem_check"))
    parser.add_argument("--out-prefix", type=Path, default=Path("reports/five_seed_completion_2026-08-10"))
    args = parser.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = read_problem_files(args.results_dir)
    summary = summarize_groups(df)
    summary.to_csv(args.out_prefix.with_suffix(".all_groups.csv"), index=False)

    ic50 = summary[(summary["dataset"].eq("chembl35_ic50_top20_v2")) & (summary["endpoint"].eq("IC50"))]
    ic50_complete = ic50[(ic50["targets"].eq(20)) & (ic50["splits"].eq(4)) & (ic50["seeds"].eq(5))].copy()
    ic50_rank = ic50_complete.sort_values(["test_censored_nll_mean", "test_mae_mean"])
    ic50_rank.to_csv(args.out_prefix.with_suffix(".ic50_rank.csv"), index=False)

    main_query = {
        "dataset": "chembl35_ic50_top20_v2",
        "endpoint": "IC50",
        "model": "lgbm",
        "feature_set": "ecfp_desc",
        "metadata_mode": "source_light",
        "treatment": "censored_likelihood",
        "censor_weight": 0.25,
        "censored_anchor_weight": 1.0,
        "right_censor_weight": 0.1,
        "right_censored_anchor_weight": 0.0,
    }
    baselines = [
        (
            "LGBM ECFP+Desc source-light censored-as-exact",
            {
                "dataset": "chembl35_ic50_top20_v2",
                "endpoint": "IC50",
                "model": "lgbm",
                "feature_set": "ecfp_desc",
                "metadata_mode": "source_light",
                "treatment": "censored_as_exact",
            },
        ),
        (
            "XGB ECFP+Desc source-light exact-only",
            {
                "dataset": "chembl35_ic50_top20_v2",
                "endpoint": "IC50",
                "model": "xgb",
                "feature_set": "ecfp_desc",
                "metadata_mode": "source_light",
                "treatment": "exact_only",
            },
        ),
        (
            "LGBM ECFP+Desc source-light pure censored",
            {
                "dataset": "chembl35_ic50_top20_v2",
                "endpoint": "IC50",
                "model": "lgbm",
                "feature_set": "ecfp_desc",
                "metadata_mode": "source_light",
                "treatment": "censored_likelihood",
                "censor_weight": 1.0,
                "censored_anchor_weight": 0.0,
            },
        ),
    ]
    pairs = pairwise(df, main_query, baselines)
    pairs.to_csv(args.out_prefix.with_suffix(".pairwise.csv"), index=False)

    endpoints = summary[
        summary["dataset"].isin(["chembl35_ic50_top20_v2", "chembl35_ki_top20_v2", "chembl35_ec50_top20_v2", "chembl35_kd_top20_v2"])
        & summary["model"].eq("xgb")
        & summary["feature_set"].eq("ecfp_desc")
        & summary["metadata_mode"].eq("source_light")
        & summary["seeds"].ge(5)
    ].copy()
    endpoints = endpoints.sort_values(["dataset", "treatment"])
    endpoints.to_csv(args.out_prefix.with_suffix(".endpoint_support.csv"), index=False)

    write_md(ic50_rank, endpoints, pairs, args.out_prefix.with_suffix(".md"))
    print(f"[done] {args.out_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
