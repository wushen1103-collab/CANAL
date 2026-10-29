#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def fmt(x: object, digits: int = 4) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)


def markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c], digits) for c in cols) + " |")
    return "\n".join(lines)


def present(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    return text.ne("") & text.str.lower().ne("none") & text.str.lower().ne("nan")


def add_repeat_stats(pred: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    keys = ["parent_inchikey", "target_chembl_id", "endpoint"]
    exact = dataset[dataset["p_relation"].eq("=")].copy()
    stats = (
        exact.groupby(keys, dropna=False)["p_activity"]
        .agg(repeat_exact_n="size", repeat_exact_min="min", repeat_exact_max="max", repeat_exact_std="std")
        .reset_index()
    )
    stats["repeat_exact_range"] = stats["repeat_exact_max"] - stats["repeat_exact_min"]
    out = pred.merge(stats, on=keys, how="left")
    out["repeat_exact_n"] = out["repeat_exact_n"].fillna(0).astype(int)
    return out


def year_bin(year: pd.Series) -> pd.Series:
    y = pd.to_numeric(year, errors="coerce")
    return pd.Series(
        np.select(
            [y.isna(), y <= 2010, y <= 2018, y > 2018],
            ["year_missing", "year_<=2010", "year_2011_2018", "year_>=2019"],
            default="year_missing",
        ),
        index=year.index,
    )


def cohort_masks(df: pd.DataFrame, min_instances: int) -> list[tuple[str, str, pd.Series]]:
    true = pd.Series(True, index=df.index)
    conf = pd.to_numeric(df["confidence_score"], errors="coerce")
    duplicate = pd.to_numeric(df["potential_duplicate"], errors="coerce").fillna(0).astype(int).eq(1)
    validity = present(df["data_validity_comment"])
    activity_comment = present(df["activity_comment"])
    repeat_n = pd.to_numeric(df["repeat_exact_n"], errors="coerce").fillna(0)
    repeat_range = pd.to_numeric(df["repeat_exact_range"], errors="coerce")
    relation = df["p_relation"].fillna("").astype(str)

    masks: list[tuple[str, str, pd.Series]] = [
        ("overall", "all", true),
        ("confidence", "confidence>=9", conf.ge(9)),
        ("confidence", "confidence=8", conf.eq(8)),
        ("confidence", "confidence<=7_or_missing", conf.le(7) | conf.isna()),
        ("validity", "potential_duplicate=1", duplicate),
        ("validity", "potential_duplicate=0", ~duplicate),
        ("validity", "data_validity_comment_present", validity),
        ("validity", "data_validity_comment_absent", ~validity),
        ("validity", "activity_comment_present", activity_comment),
        ("validity", "activity_comment_absent", ~activity_comment),
        ("repeat_exact", "repeat_exact_n>=2", repeat_n.ge(2)),
        ("repeat_exact", "repeat_exact_range>=0.5", repeat_range.ge(0.5)),
        ("repeat_exact", "repeat_exact_range>=1.0", repeat_range.ge(1.0)),
        ("relation", "censored_relation_<", relation.eq("<")),
        ("relation", "censored_relation_>", relation.eq(">")),
    ]

    for split in sorted(df["split"].dropna().unique()):
        masks.append(("split", f"split={split}", df["split"].eq(split)))

    assay_counts = df["assay_type"].fillna("missing").astype(str).value_counts()
    for assay_type in assay_counts.head(8).index:
        mask = df["assay_type"].fillna("missing").astype(str).eq(assay_type)
        masks.append(("assay_type", f"assay_type={assay_type}", mask))

    bins = year_bin(df["document_year"])
    for label in ["year_<=2010", "year_2011_2018", "year_>=2019", "year_missing"]:
        masks.append(("year", label, bins.eq(label)))

    return [(kind, name, mask) for kind, name, mask in masks if int(mask.sum()) >= min_instances]


def summarize_subset(df: pd.DataFrame, cohort_type: str, cohort: str) -> list[dict[str, object]]:
    rows = []
    for (treatment, method), group in df.groupby(["treatment", "method"], dropna=False):
        exact = group["is_exact"].fillna(False).astype(bool)
        censored = ~exact
        row = {
            "cohort_type": cohort_type,
            "cohort": cohort,
            "treatment": treatment,
            "method": method,
            "instances": len(group),
            "unique_rows": group["row_id"].nunique(),
            "exact_instances": int(exact.sum()),
            "censored_instances": int(censored.sum()),
            "mae": pd.to_numeric(group.loc[exact, "abs_error"], errors="coerce").mean(),
            "gaussian_nll": pd.to_numeric(group.loc[exact, "gaussian_nll"], errors="coerce").mean(),
            "censored_nll": pd.to_numeric(group.loc[censored, "censored_nll"], errors="coerce").mean(),
            "censored_dir_acc": pd.to_numeric(group.loc[censored, "censored_dir_ok"], errors="coerce").mean(),
            "sigma_mean": pd.to_numeric(group["sigma"], errors="coerce").mean(),
            "target_count": group["target_chembl_id"].nunique(),
            "split_count": group["split"].nunique(),
        }
        rows.append(row)
    return rows


def add_deltas(summary: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = summary.copy()
    baselines = {"exact": "exact_only", "naive": "censored_as_exact"}
    metrics = ["mae", "gaussian_nll", "censored_nll", "censored_dir_acc"]
    for label, treatment in baselines.items():
        base = out[out["treatment"].eq(treatment)][keys + metrics].copy()
        base = base.rename(columns={metric: f"{metric}_{label}" for metric in metrics})
        out = out.merge(base, on=keys, how="left")
        for metric in metrics:
            out[f"{metric}_delta_vs_{label}"] = out[metric] - out[f"{metric}_{label}"]
        out = out.drop(columns=[f"{metric}_{label}" for metric in metrics])
    return out


def summarize_cohorts(pred: pd.DataFrame, min_instances: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort_type, cohort, mask in cohort_masks(pred, min_instances):
        rows.extend(summarize_subset(pred.loc[mask].copy(), cohort_type, cohort))
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary = add_deltas(summary, ["cohort_type", "cohort"])
    order = {
        "overall": 0,
        "split": 1,
        "confidence": 2,
        "validity": 3,
        "repeat_exact": 4,
        "relation": 5,
        "assay_type": 6,
        "year": 7,
    }
    summary["_order"] = summary["cohort_type"].map(order).fillna(99)
    return summary.sort_values(["_order", "cohort", "treatment"]).drop(columns=["_order"])


def summarize_targets(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, split), group in pred.groupby(["target_chembl_id", "split"], dropna=False):
        for row in summarize_subset(group.copy(), "target_split", f"{target}:{split}"):
            row["target_chembl_id"] = target
            row["split"] = split
            rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return add_deltas(summary, ["target_chembl_id", "split", "cohort_type", "cohort"])


def metric(summary: pd.DataFrame, cohort: str, treatment: str, column: str) -> float:
    row = summary[(summary["cohort"].eq(cohort)) & (summary["treatment"].eq(treatment))]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def build_markdown(pred: pd.DataFrame, cohort_summary: pd.DataFrame, target_summary: pd.DataFrame, pred_path: Path) -> str:
    overall = cohort_summary[cohort_summary["cohort"].eq("all")].copy()
    overall_cols = [
        "method",
        "instances",
        "exact_instances",
        "censored_instances",
        "mae",
        "censored_nll",
        "censored_dir_acc",
        "mae_delta_vs_exact",
        "censored_nll_delta_vs_exact",
        "censored_nll_delta_vs_naive",
        "censored_dir_acc_delta_vs_exact",
    ]

    selected = [
        "all",
        "confidence>=9",
        "confidence=8",
        "potential_duplicate=1",
        "data_validity_comment_present",
        "repeat_exact_range>=0.5",
        "repeat_exact_range>=1.0",
        "censored_relation_<",
        "censored_relation_>",
        "assay_type=B",
        "assay_type=F",
    ]
    selected_rows = cohort_summary[
        cohort_summary["cohort"].isin(selected) & cohort_summary["treatment"].eq("censored_likelihood")
    ].copy()
    selected_cols = [
        "cohort",
        "instances",
        "exact_instances",
        "censored_instances",
        "mae",
        "censored_nll",
        "censored_dir_acc",
        "mae_delta_vs_exact",
        "censored_nll_delta_vs_exact",
        "censored_nll_delta_vs_naive",
        "censored_dir_acc_delta_vs_exact",
    ]

    target_canal = target_summary[target_summary["treatment"].eq("censored_likelihood")].copy()
    best_targets = target_canal.sort_values("censored_nll_delta_vs_naive").head(8)
    weak_targets = target_canal.sort_values("censored_nll_delta_vs_naive", ascending=False).head(8)
    target_cols = [
        "target_chembl_id",
        "split",
        "instances",
        "censored_instances",
        "mae_delta_vs_exact",
        "censored_nll_delta_vs_naive",
        "censored_dir_acc_delta_vs_naive",
    ]

    canal_cnll = metric(cohort_summary, "all", "censored_likelihood", "censored_nll")
    exact_cnll = metric(cohort_summary, "all", "exact_only", "censored_nll")
    naive_cnll = metric(cohort_summary, "all", "censored_as_exact", "censored_nll")
    canal_dir = metric(cohort_summary, "all", "censored_likelihood", "censored_dir_acc")
    exact_dir = metric(cohort_summary, "all", "exact_only", "censored_dir_acc")
    canal_method = cohort_summary[
        cohort_summary["cohort"].eq("all") & cohort_summary["treatment"].eq("censored_likelihood")
    ]["method"]
    canal_method_name = str(canal_method.iloc[0]) if len(canal_method) else "CANAL"
    canal_mae_delta = metric(cohort_summary, "all", "censored_likelihood", "mae_delta_vs_exact")
    repeat_mae = metric(cohort_summary, "repeat_exact_range>=1.0", "exact_only", "mae")
    all_exact_mae = metric(cohort_summary, "all", "exact_only", "mae")
    repeat_cnll_delta = metric(cohort_summary, "repeat_exact_range>=1.0", "censored_likelihood", "censored_nll_delta_vs_naive")

    lines = [
        "# Top-20 IC50 Prediction Noise Diagnostics",
        "",
        f"Input predictions: `{pred_path}`",
        f"Prediction instances: {len(pred):,}; unique dataset rows touched: {pred['row_id'].nunique():,}.",
        "",
        "## Overall Method Metrics",
        "",
        markdown_table(overall[overall_cols]),
        "",
        "## CANAL Cohort Deltas",
        "",
        markdown_table(selected_rows[selected_cols]),
        "",
        "## Target/Split Wins Against Naive",
        "",
        "Strongest CANAL censored-NLL gains vs censored-as-exact:",
        "",
        markdown_table(best_targets[target_cols]),
        "",
        "Weakest CANAL censored-NLL cases vs censored-as-exact:",
        "",
        markdown_table(weak_targets[target_cols]),
        "",
        "## Key Findings",
        "",
        (
            f"1. Overall, {canal_method_name} has censored NLL {canal_cnll:.4f}, "
            f"vs exact-only {exact_cnll:.4f} and censored-as-exact {naive_cnll:.4f}; "
            f"direction accuracy is {canal_dir:.4f} vs exact-only {exact_dir:.4f}."
        ),
        (
            f"2. The exact-value cost in this seed-level prediction export is MAE delta "
            f"{canal_mae_delta:+.4f} vs exact-only, so the main tradeoff remains calibration/noise handling "
            "rather than raw exact regression."
        ),
        (
            f"3. Repeated exact measurements with range >=1.0 have exact-only MAE {repeat_mae:.4f} "
            f"vs overall exact-only MAE {all_exact_mae:.4f}; CANAL's censored-NLL delta vs naive in that cohort is "
            f"{repeat_cnll_delta:+.4f}."
        ),
        "4. The target/split table should be used to identify where CANAL loses to the naive threshold-as-value baseline before deciding any further hyperparameter change.",
        "",
        "## Outputs",
        "",
        "- Cohort summary CSV: sibling file ending `_cohort_summary.csv`.",
        "- Target/split summary CSV: sibling file ending `_target_summary.csv`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--min-instances", type=int, default=100)
    args = parser.parse_args()

    pred = pd.read_parquet(args.predictions)
    dataset = pd.read_parquet(args.dataset).reset_index(drop=True)
    pred = add_repeat_stats(pred, dataset)

    cohort_summary = summarize_cohorts(pred, args.min_instances)
    target_summary = summarize_targets(pred)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cohort_path = args.out_prefix.with_name(args.out_prefix.name + "_cohort_summary.csv")
    target_path = args.out_prefix.with_name(args.out_prefix.name + "_target_summary.csv")
    md_path = args.out_prefix.with_suffix(".md")
    cohort_summary.to_csv(cohort_path, index=False)
    target_summary.to_csv(target_path, index=False)
    md_path.write_text(build_markdown(pred, cohort_summary, target_summary, args.predictions), encoding="utf-8")

    print(f"[done] wrote {cohort_path}")
    print(f"[done] wrote {target_path}")
    print(f"[done] wrote {md_path}")


if __name__ == "__main__":
    main()
