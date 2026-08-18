#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


METRIC_COLS = [
    "mae_exact",
    "rmse_exact",
    "spearman_exact",
    "censored_nll",
    "boundary_satisfaction",
    "interval_miss_rate",
    "sigma",
    "latent_mae",
    "latent_rmse",
    "latent_spearman",
    "latent_mae_all",
    "latent_rmse_all",
    "latent_spearman_all",
    "latent_calib_50",
    "latent_calib_90",
]


def infer_dataset(path: Path) -> str:
    text = str(path).lower()
    if "bindingdb" in text and "independent" in text:
        return "BindingDB-IndependentStrict-202608"
    if "bindingdb" in text:
        return "BindingDB-Articles-202608"
    if "chembl" in text:
        return "ChEMBL37-CANAL20"
    return "unknown"


def infer_experiment(path: Path) -> str:
    text = str(path).lower()
    if "meta_perm" in text:
        return "metadata_permuted_control"
    if "bindingdb" in text and "independent" in text and "calibrated" in text:
        return "external_bindingdb_independent_calibrated"
    if "bindingdb" in text and "independent" in text:
        return "external_bindingdb_independent"
    if "bindingdb" in text and "calibrated" in text:
        return "external_bindingdb_calibrated"
    if "bindingdb" in text:
        return "external_bindingdb"
    if "calibrated" in text:
        return "canal_calibrated_backbone"
    if "tree" in text:
        return "tree_baselines"
    if "smiles_cnn" in text:
        return "smiles_cnn_baseline"
    if "core" in text:
        return "core"
    return path.name


def load_one_dir(path: Path) -> pd.DataFrame:
    path = Path(path)
    if (path / "metrics_long.csv").exists():
        df = pd.read_csv(path / "metrics_long.csv")
    else:
        files = sorted(p for p in path.glob("metrics_*.csv") if p.name != "metrics_long.csv")
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    if df.empty:
        return df
    df["artifact_dir"] = str(path)
    df["dataset"] = infer_dataset(path)
    df["experiment"] = infer_experiment(path)
    return df


def fmt_mean_std(mean: float, std: float, count: int, digits: int = 4) -> str:
    if not np.isfinite(mean):
        return ""
    if not np.isfinite(std) or count <= 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def aggregate(metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    value_cols = [c for c in METRIC_COLS if c in metrics.columns]
    agg = metrics.groupby(group_cols, dropna=False)[value_cols].agg(["mean", "std", "count"]).reset_index()
    flat_cols = []
    for col in agg.columns:
        if isinstance(col, tuple):
            flat_cols.append("_".join([str(x) for x in col if str(x)]).strip("_"))
        else:
            flat_cols.append(str(col))
    agg.columns = flat_cols
    return agg


def make_pretty_table(agg: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for _, row in agg.iterrows():
        out = {c: row[c] for c in group_cols if c in row.index}
        for metric in METRIC_COLS:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            count_col = f"{metric}_count"
            if mean_col in row.index:
                out[metric] = fmt_mean_std(float(row[mean_col]), float(row.get(std_col, np.nan)), int(row.get(count_col, 0)))
        rows.append(out)
    return pd.DataFrame(rows)


def paired_deltas(metrics: pd.DataFrame, baselines: list[str]) -> pd.DataFrame:
    value_cols = [c for c in METRIC_COLS if c in metrics.columns]
    id_cols = ["dataset", "experiment", "semisynthetic", "split", "seed"]
    rows = []
    for baseline in baselines:
        base = metrics[metrics["method"].eq(baseline)][id_cols + value_cols].rename(columns={c: f"{c}_baseline" for c in value_cols})
        if base.empty:
            continue
        merged = metrics.merge(base, on=id_cols, how="inner")
        merged = merged[merged["method"].ne(baseline)].copy()
        for metric in value_cols:
            merged[f"delta_{metric}_vs_{baseline}"] = merged[metric] - merged[f"{metric}_baseline"]
        delta_cols = [c for c in merged.columns if c.startswith("delta_")]
        if not delta_cols:
            continue
        agg = merged.groupby(["dataset", "experiment", "semisynthetic", "method"], dropna=False)[delta_cols].agg(["mean", "std", "count"]).reset_index()
        agg.columns = ["_".join([str(x) for x in c if str(x)]).strip("_") if isinstance(c, tuple) else str(c) for c in agg.columns]
        agg.insert(4, "baseline", baseline)
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def ranking_distortion(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    required = {"dataset", "experiment", "semisynthetic", "split", "seed", "method", "mae_exact", "censored_nll"}
    if not required.issubset(metrics.columns):
        return pd.DataFrame()
    for (dataset, experiment, semi, split, seed), g in metrics.dropna(subset=["mae_exact", "censored_nll"]).groupby(["dataset", "experiment", "semisynthetic", "split", "seed"]):
        if g["method"].nunique() < 2:
            continue
        mae_rank = g["mae_exact"].rank(method="average", ascending=True)
        nll_rank = g["censored_nll"].rank(method="average", ascending=True)
        tau = kendalltau(mae_rank, nll_rank).statistic
        rows.append(
            {
                "dataset": dataset,
                "experiment": experiment,
                "semisynthetic": bool(semi),
                "split": split,
                "seed": int(seed),
                "methods": int(g["method"].nunique()),
                "top_mae": str(g.loc[g["mae_exact"].idxmin(), "method"]),
                "top_nll": str(g.loc[g["censored_nll"].idxmin(), "method"]),
                "top1_changed": bool(g.loc[g["mae_exact"].idxmin(), "method"] != g.loc[g["censored_nll"].idxmin(), "method"]),
                "kendall_rank_corr": float(tau) if np.isfinite(tau) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def markdown_top(agg_pretty: pd.DataFrame, semisynthetic: bool, metric: str, title: str) -> str:
    df = agg_pretty[agg_pretty["semisynthetic"].eq(semisynthetic)].copy()
    if df.empty or metric not in df.columns:
        return f"## {title}\n\nNo rows.\n"
    raw_metric_col = f"{metric}_mean"
    # The pretty table no longer has numeric means; caller uses unpretty table for ordering if needed.
    return f"## {title}\n\n" + df.to_markdown(index=False) + "\n"


def to_markdown_simple(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    cols = list(df.columns)
    def cell(x: object) -> str:
        if pd.isna(x):
            return ""
        return str(x).replace("\n", " ").replace("|", "\\|")
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baselines", nargs="+", default=["ridge_molecule_only_naive", "tobit_mlp_molecule"])
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [load_one_dir(Path(p)) for p in args.inputs]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("No metric files found.")
    metrics = pd.concat(frames, ignore_index=True)
    # A tree/core directory may have been summarized from per-case files and then combined again;
    # keep one row per natural case-method-artifact.
    metrics = metrics.drop_duplicates(subset=["dataset", "experiment", "semisynthetic", "split", "seed", "method", "artifact_dir"])
    metrics.to_csv(out_dir / "metrics_long.csv", index=False)

    group_overall = ["dataset", "experiment", "semisynthetic", "method"]
    group_split = ["dataset", "experiment", "semisynthetic", "split", "method"]
    overall = aggregate(metrics, group_overall)
    by_split = aggregate(metrics, group_split)
    overall.to_csv(out_dir / "metric_mean_std_overall_by_method.csv", index=False)
    by_split.to_csv(out_dir / "metric_mean_std_by_split_method.csv", index=False)
    make_pretty_table(overall, group_overall).to_csv(out_dir / "metric_mean_std_overall_by_method.pretty.csv", index=False)
    make_pretty_table(by_split, group_split).to_csv(out_dir / "metric_mean_std_by_split_method.pretty.csv", index=False)

    deltas = paired_deltas(metrics, args.baselines)
    if not deltas.empty:
        deltas.to_csv(out_dir / "paired_deltas_vs_baselines.csv", index=False)

    ranks = ranking_distortion(metrics)
    if not ranks.empty:
        ranks.to_csv(out_dir / "ranking_distortion.csv", index=False)
        rank_summary = ranks.groupby(["dataset", "experiment", "semisynthetic"], dropna=False).agg(
            cases=("top1_changed", "count"),
            top1_changed_rate=("top1_changed", "mean"),
            kendall_rank_corr_mean=("kendall_rank_corr", "mean"),
            kendall_rank_corr_std=("kendall_rank_corr", "std"),
        )
        rank_summary.reset_index().to_csv(out_dir / "ranking_distortion_summary.csv", index=False)

    pretty = make_pretty_table(overall, group_overall)
    (out_dir / "overall_real.md").write_text(
        to_markdown_simple(pretty[pretty["semisynthetic"].eq(False)]),
        encoding="utf-8",
    )
    (out_dir / "overall_semisynthetic.md").write_text(
        to_markdown_simple(pretty[pretty["semisynthetic"].eq(True)]),
        encoding="utf-8",
    )
    manifest = {
        "inputs": args.inputs,
        "rows": int(len(metrics)),
        "datasets": sorted(metrics["dataset"].dropna().unique().tolist()),
        "experiments": sorted(metrics["experiment"].dropna().unique().tolist()),
        "methods": sorted(metrics["method"].dropna().unique().tolist()),
        "splits": sorted(metrics["split"].dropna().unique().tolist()),
        "seeds": sorted(int(x) for x in metrics["seed"].dropna().unique().tolist()),
    }
    (out_dir / "combined_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
