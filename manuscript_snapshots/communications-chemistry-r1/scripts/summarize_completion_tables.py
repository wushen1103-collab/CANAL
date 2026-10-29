#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def fmt_mean_std(mean: float, std: float, digits: int = 4) -> str:
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def fmt_config_value(value: object) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(num) or num == -1.0:
        return ""
    return f"{num:g}"


def censored_method_label(row: pd.Series) -> str:
    parts = [
        str(row.get("model", "")).upper(),
        str(row.get("feature_set", "")),
        str(row.get("metadata_mode", "")),
        str(row.get("treatment", "")),
    ]
    if row.get("treatment") == "censored_likelihood":
        config = []
        for col, label in [
            ("censor_weight", "cw"),
            ("censored_anchor_weight", "aw"),
            ("right_censor_weight", "right_cw"),
            ("right_censored_anchor_weight", "right_aw"),
        ]:
            value = fmt_config_value(row.get(col, np.nan))
            if value:
                config.append(f"{label}={value}")
        if config:
            parts.append("(" + ", ".join(config) + ")")
    sigma = row.get("sigma_calibration", "")
    if isinstance(sigma, str) and sigma and sigma != "exact":
        parts.append(f"sigma={sigma}")
    return " ".join(part for part in parts if part)


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return ""
    rows = df[cols].astype(str).values.tolist()
    widths = [len(col) for col in cols]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    header = "| " + " | ".join(col.ljust(width) for col, width in zip(cols, widths)) + " |"
    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def summarize_censored(files: list[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        if path.exists():
            frame = pd.read_csv(path)
            frame["source_file"] = path.name
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    ok = df[df["status"].eq("ok")].copy()
    for col, default in [
        ("dataset", "unknown"),
        ("model", "unknown"),
        ("feature_set", "ecfp"),
        ("metadata_mode", "none"),
    ]:
        if col not in ok.columns:
            ok[col] = default
        else:
            ok[col] = ok[col].fillna(default)
    group_cols = ["dataset", "endpoint", "split", "treatment", "model", "feature_set", "metadata_mode"]
    for col, default in [
        ("sigma_calibration", "exact"),
        ("censor_weight", -1.0),
        ("censored_anchor_weight", -1.0),
        ("right_censor_weight", -1.0),
        ("right_censored_anchor_weight", -1.0),
        ("posthoc_bias", "none"),
        ("posthoc_lambda", -1.0),
    ]:
        if col in ok.columns:
            ok[col] = ok[col].fillna(default)
            group_cols.append(col)
    rows = []
    for keys, group in ok.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["runs"] = len(group)
        for metric in ["test_mae", "test_censored_nll", "test_censored_dir_acc", "worst_assay_mae"]:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def summarize_tdc(files: list[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        if path.exists():
            frame = pd.read_csv(path)
            frame["source_file"] = path.name
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    rows = []
    group_cols = ["dataset", "split", "model", "feature_set", "label_space"]
    for keys, group in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["runs"] = len(group)
        row["drug_overlap"] = int(group["train_test_drug_overlap"].max())
        row["target_overlap"] = int(group["train_test_target_overlap"].max())
        for metric in ["test_mae", "test_rmse", "test_spearman", "test_pearson", "test_cindex_sampled"]:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def write_markdown(censored: pd.DataFrame, tdc: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Baseline Completion Tables")
    lines.append("")
    lines.append("All rows in these tables are rerun in this repository. Literature-only SOTA references are not mixed into these numbers.")
    lines.append("")
    if not censored.empty:
        lines.append("## ChEMBL Endpoint Panels")
        show = censored.copy()
        show["method"] = show.apply(censored_method_label, axis=1)
        show["MAE"] = [fmt_mean_std(m, s) for m, s in zip(show["test_mae_mean"], show["test_mae_std"])]
        show["censored NLL"] = [
            fmt_mean_std(m, s) for m, s in zip(show["test_censored_nll_mean"], show["test_censored_nll_std"])
        ]
        show["censored dir"] = [
            fmt_mean_std(m, s) for m, s in zip(show["test_censored_dir_acc_mean"], show["test_censored_dir_acc_std"])
        ]
        show["worst assay MAE"] = [
            fmt_mean_std(m, s) for m, s in zip(show["worst_assay_mae_mean"], show["worst_assay_mae_std"])
        ]
        cols = [
            "dataset",
            "endpoint",
            "split",
            "method",
            "runs",
            "MAE",
            "censored NLL",
            "censored dir",
            "worst assay MAE",
        ]
        lines.append(markdown_table(show, cols))
        lines.append("")
    if not tdc.empty:
        lines.append("## TDC External DTI")
        show = tdc.copy()
        show["MAE"] = [fmt_mean_std(m, s) for m, s in zip(show["test_mae_mean"], show["test_mae_std"])]
        show["RMSE"] = [fmt_mean_std(m, s) for m, s in zip(show["test_rmse_mean"], show["test_rmse_std"])]
        show["Spearman"] = [
            fmt_mean_std(m, s) for m, s in zip(show["test_spearman_mean"], show["test_spearman_std"])
        ]
        show["Pearson"] = [fmt_mean_std(m, s) for m, s in zip(show["test_pearson_mean"], show["test_pearson_std"])]
        show["sampled C-index"] = [
            fmt_mean_std(m, s) for m, s in zip(show["test_cindex_sampled_mean"], show["test_cindex_sampled_std"])
        ]
        cols = [
            "dataset",
            "split",
            "model",
            "runs",
            "drug_overlap",
            "target_overlap",
            "MAE",
            "RMSE",
            "Spearman",
            "Pearson",
            "sampled C-index",
        ]
        lines.append(markdown_table(show, cols))
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--censored", nargs="*", type=Path, default=[])
    parser.add_argument("--tdc", nargs="*", type=Path, default=[])
    parser.add_argument("--out-prefix", type=Path, default=Path("reports/baseline_completion_tables"))
    args = parser.parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    censored = summarize_censored(args.censored)
    tdc = summarize_tdc(args.tdc)
    if not censored.empty:
        censored.to_csv(args.out_prefix.with_suffix(".censored.csv"), index=False)
    if not tdc.empty:
        tdc.to_csv(args.out_prefix.with_suffix(".tdc.csv"), index=False)
    write_markdown(censored, tdc, args.out_prefix.with_suffix(".md"))
    print(f"[done] wrote {args.out_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
