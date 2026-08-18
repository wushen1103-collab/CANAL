#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


MAIN_EXPERIMENTS = {"core", "tree_baselines", "smiles_cnn_baseline", "canal_calibrated_backbone"}
EXTERNAL_EXPERIMENTS = {"external_bindingdb", "external_bindingdb_calibrated"}
PRIMARY_METHOD = "canal_extratrees_molecule_calibrated"
BASELINE_METHOD = "extratrees_molecule_only_naive"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def norm_doi(x: object) -> str:
    s = str(x or "").strip().lower()
    if not s or s in {"nan", "none"}:
        return ""
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    s = s.strip().strip(".")
    return s


def norm_pmid(x: object) -> str:
    s = str(x or "").strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    m = re.search(r"\d+", s)
    return m.group(0) if m else ""


def bool_rate(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def overlap_audit(chembl_path: Path, bindingdb_path: Path, out_dir: Path) -> dict:
    chembl = pd.read_parquet(chembl_path)
    bdb = pd.read_parquet(bindingdb_path)
    c = chembl.copy()
    b = bdb.copy()
    for df in (c, b):
        for col in ["molecule_key", "canonical_smiles_rdkit", "uniprot_accession", "endpoint", "p_activity_threshold"]:
            if col not in df.columns:
                df[col] = ""
    c["doi_norm"] = c.get("doi", pd.Series([""] * len(c))).map(norm_doi)
    b["doi_norm"] = b.get("doi", pd.Series([""] * len(b))).map(norm_doi)
    c["pmid_norm"] = c.get("pubmed_id", pd.Series([""] * len(c))).map(norm_pmid)
    b["pmid_norm"] = b.get("pubmed_id", pd.Series([""] * len(b))).map(norm_pmid)

    c_mte = set(map(tuple, c[["molecule_key", "uniprot_accession", "endpoint"]].fillna("").astype(str).to_numpy()))
    c_ct = set(map(tuple, c[["canonical_smiles_rdkit", "uniprot_accession"]].fillna("").astype(str).to_numpy()))
    c_mt = set(map(tuple, c[["molecule_key", "uniprot_accession"]].fillna("").astype(str).to_numpy()))
    c_doi = {x for x in c["doi_norm"].dropna().astype(str) if x}
    c_pmid = {x for x in c["pmid_norm"].dropna().astype(str) if x}
    c_mol = set(c["molecule_key"].dropna().astype(str))
    c_target = set(c["uniprot_accession"].dropna().astype(str))

    b["overlap_exact_molecule_target_endpoint"] = [key in c_mte for key in map(tuple, b[["molecule_key", "uniprot_accession", "endpoint"]].fillna("").astype(str).to_numpy())]
    b["overlap_canonical_molecule_target"] = [key in c_ct for key in map(tuple, b[["canonical_smiles_rdkit", "uniprot_accession"]].fillna("").astype(str).to_numpy())]
    b["overlap_molecule_target_no_endpoint"] = [key in c_mt for key in map(tuple, b[["molecule_key", "uniprot_accession"]].fillna("").astype(str).to_numpy())]
    b["overlap_molecule_only"] = b["molecule_key"].fillna("").astype(str).isin(c_mol)
    b["overlap_target_only"] = b["uniprot_accession"].fillna("").astype(str).isin(c_target)
    b["overlap_doi"] = b["doi_norm"].astype(str).isin(c_doi) & b["doi_norm"].astype(str).ne("")
    b["overlap_pmid"] = b["pmid_norm"].astype(str).isin(c_pmid) & b["pmid_norm"].astype(str).ne("")
    b["overlap_article"] = b["overlap_doi"] | b["overlap_pmid"]

    # Near measurement overlap: same InChIKey/UniProt/endpoint and close p-activity threshold.
    value_map: dict[tuple[str, str, str], np.ndarray] = {}
    for key, g in c.groupby(["molecule_key", "uniprot_accession", "endpoint"], dropna=False):
        vals = pd.to_numeric(g["p_activity_threshold"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals):
            value_map[tuple(str(x) for x in key)] = np.sort(vals)

    near_001 = []
    near_005 = []
    min_abs_delta = []
    for _, row in b.iterrows():
        key = (str(row.get("molecule_key", "")), str(row.get("uniprot_accession", "")), str(row.get("endpoint", "")))
        vals = value_map.get(key)
        y = pd.to_numeric(pd.Series([row.get("p_activity_threshold", np.nan)]), errors="coerce").iloc[0]
        if vals is None or not np.isfinite(y):
            near_001.append(False)
            near_005.append(False)
            min_abs_delta.append(np.nan)
            continue
        delta = float(np.min(np.abs(vals - y)))
        min_abs_delta.append(delta)
        near_001.append(delta <= 0.01)
        near_005.append(delta <= 0.05)
    b["overlap_measurement_pactivity_le_0p01"] = near_001
    b["overlap_measurement_pactivity_le_0p05"] = near_005
    b["overlap_measurement_min_abs_delta"] = min_abs_delta

    b["remove_article_independent"] = b["overlap_article"]
    b["remove_strict_independent"] = (
        b["overlap_article"]
        | b["overlap_exact_molecule_target_endpoint"]
        | b["overlap_canonical_molecule_target"]
        | b["overlap_measurement_pactivity_le_0p05"]
    )

    flags = [
        "overlap_exact_molecule_target_endpoint",
        "overlap_canonical_molecule_target",
        "overlap_molecule_target_no_endpoint",
        "overlap_molecule_only",
        "overlap_target_only",
        "overlap_doi",
        "overlap_pmid",
        "overlap_article",
        "overlap_measurement_pactivity_le_0p01",
        "overlap_measurement_pactivity_le_0p05",
    ]
    summary_rows = []
    for flag in flags:
        summary_rows.append({"flag": flag, "rows": int(b[flag].sum()), "rate": bool_rate(b[flag]), "unique_target_endpoints": int(b.loc[b[flag], "target_endpoint"].nunique())})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "bindingdb_overlap_summary.csv", index=False)

    article_ind = b[~b["remove_article_independent"]].drop(columns=[c for c in b.columns if c.startswith("overlap_") or c.startswith("remove_") or c in {"doi_norm", "pmid_norm"}]).copy()
    strict_ind = b[~b["remove_strict_independent"]].drop(columns=[c for c in b.columns if c.startswith("overlap_") or c.startswith("remove_") or c in {"doi_norm", "pmid_norm"}]).copy()
    article_path = out_dir.parent.parent / "data" / "processed" / "bindingdb_articles_202608_article_independent.parquet"
    strict_path = out_dir.parent.parent / "data" / "processed" / "bindingdb_articles_202608_independent_strict.parquet"
    ensure_dir(article_path.parent)
    article_ind.to_parquet(article_path, index=False)
    strict_ind.to_parquet(strict_path, index=False)
    b[["activity_id", "target_endpoint", "molecule_key", "canonical_smiles_rdkit", "uniprot_accession", "endpoint", "p_activity_threshold", "doi", "pubmed_id"] + flags + ["overlap_measurement_min_abs_delta", "remove_article_independent", "remove_strict_independent"]].to_csv(out_dir / "bindingdb_overlap_row_flags.csv", index=False)

    def panel_manifest(df: pd.DataFrame, name: str, path: Path) -> dict:
        return {
            "name": name,
            "path": str(path),
            "rows": int(len(df)),
            "target_endpoints": int(df["target_endpoint"].nunique()),
            "targets": int(df["target_chembl_id"].nunique()) if "target_chembl_id" in df.columns else int(df["uniprot_accession"].nunique()),
            "molecules": int(df["molecule_key"].nunique()),
            "scaffolds": int(df["bemis_murcko_scaffold"].nunique()) if "bemis_murcko_scaffold" in df.columns else None,
            "docs": int(df["document_chembl_id"].nunique()) if "document_chembl_id" in df.columns else None,
            "censor_rate": float(df["is_censored"].mean()) if len(df) else None,
            "relation_counts": df["standard_relation"].value_counts(dropna=False).to_dict() if "standard_relation" in df.columns else {},
            "years": [int(df["publication_year"].min()), int(df["publication_year"].max())] if len(df) and "publication_year" in df.columns else [],
        }

    summary = {
        "chembl_rows": int(len(c)),
        "bindingdb_all": panel_manifest(bdb, "BindingDB-All", bindingdb_path),
        "bindingdb_article_independent": panel_manifest(article_ind, "BindingDB-ArticleIndependent", article_path),
        "bindingdb_independent_strict": panel_manifest(strict_ind, "BindingDB-IndependentStrict", strict_path),
        "overlap_flags": summary_rows,
    }
    write_json(out_dir / "bindingdb_independence_manifest.json", summary)
    write_json(article_path.with_suffix(".manifest.json"), summary["bindingdb_article_independent"])
    write_json(strict_path.with_suffix(".manifest.json"), summary["bindingdb_independent_strict"])
    return summary


def run_accounting(metrics_path: Path, out_dir: Path) -> dict:
    metrics = pd.read_csv(metrics_path)
    group_cols = ["dataset", "experiment", "semisynthetic"]
    breakdown = (
        metrics.groupby(group_cols)
        .agg(
            metric_records=("method", "count"),
            methods=("method", "nunique"),
            splits=("split", "nunique"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    breakdown["expected_records_from_observed_axes"] = breakdown["methods"] * breakdown["splits"] * breakdown["seeds"]
    breakdown.to_csv(out_dir / "run_accounting_breakdown.csv", index=False)
    main = metrics[metrics["dataset"].eq("ChEMBL37-CANAL20") & metrics["experiment"].isin(MAIN_EXPERIMENTS)]
    method_count = int(main["method"].nunique())
    split_count = int(main["split"].nunique())
    seed_count = int(main["seed"].nunique())
    # ChEMBL main panel has 20 target-endpoints; metrics rows aggregate across these targets.
    target_endpoints = 20
    accounting = {
        "final_metrics_rows": int(len(metrics)),
        "definition": "one metric row = one dataset/experiment/semisynthetic/split/seed/method aggregate over all target-endpoints in that panel",
        "not_definition": "1,350 is not target-level method-target-split-seed rows",
        "main_chembl_methods": method_count,
        "main_chembl_splits": split_count,
        "main_chembl_seeds": seed_count,
        "main_chembl_semisynthetic_states": int(main["semisynthetic"].nunique()),
        "main_chembl_aggregate_metric_rows": int(len(main)),
        "main_chembl_real_aggregate_metric_rows": int((~main["semisynthetic"].astype(bool)).sum()),
        "main_chembl_implied_target_method_split_seed_cells_real": int(target_endpoints * split_count * seed_count * method_count),
        "main_chembl_implied_target_method_split_seed_cells_real_plus_semi": int(target_endpoints * split_count * seed_count * method_count * int(main["semisynthetic"].nunique())),
        "breakdown_csv": str(out_dir / "run_accounting_breakdown.csv"),
    }
    write_json(out_dir / "run_accounting_manifest.json", accounting)
    return accounting


def ranking_reversal(metrics_path: Path, out_dir: Path) -> dict:
    metrics = pd.read_csv(metrics_path)
    main = metrics[
        metrics["dataset"].eq("ChEMBL37-CANAL20")
        & metrics["experiment"].isin(MAIN_EXPERIMENTS)
        & (~metrics["semisynthetic"].astype(bool))
    ].copy()
    rows_metric = []
    for (split, seed), g in main.groupby(["split", "seed"]):
        if g["method"].nunique() < 3:
            continue
        rank_mae = g.set_index("method")["mae_exact"].rank(ascending=True)
        rank_nll = g.set_index("method")["censored_nll"].rank(ascending=True)
        common = rank_mae.index.intersection(rank_nll.index)
        tau = stats.kendalltau(rank_mae.loc[common], rank_nll.loc[common]).statistic
        top_mae = rank_mae.sort_values().index.tolist()
        top_nll = rank_nll.sort_values().index.tolist()
        rows_metric.append(
            {
                "split": split,
                "seed": int(seed),
                "tau_mae_vs_nll": float(tau) if np.isfinite(tau) else np.nan,
                "top1_mae": top_mae[0],
                "top1_nll": top_nll[0],
                "top1_switch": bool(top_mae[0] != top_nll[0]),
                "top3_overlap": int(len(set(top_mae[:3]) & set(top_nll[:3]))),
                "top3_jaccard": float(len(set(top_mae[:3]) & set(top_nll[:3])) / len(set(top_mae[:3]) | set(top_nll[:3]))),
            }
        )
    metric_df = pd.DataFrame(rows_metric)
    metric_df.to_csv(out_dir / "ranking_reversal_mae_vs_nll.csv", index=False)

    rows_shift = []
    for metric, ascending in [("mae_exact", True), ("censored_nll", True), ("spearman_exact", False)]:
        avg = main.groupby(["split", "method"])[metric].mean().reset_index()
        piv = avg.pivot(index="method", columns="split", values=metric)
        for a in sorted(piv.columns):
            for b in sorted(piv.columns):
                if a >= b:
                    continue
                common = piv[[a, b]].dropna()
                if len(common) < 3:
                    continue
                # rank 1 = best for both lower-is-better and higher-is-better metrics.
                ra = common[a].rank(ascending=ascending)
                rb = common[b].rank(ascending=ascending)
                tau = stats.kendalltau(ra, rb).statistic
                top_a = common[a].sort_values(ascending=ascending).index.tolist()
                top_b = common[b].sort_values(ascending=ascending).index.tolist()
                rows_shift.append(
                    {
                        "metric": metric,
                        "split_a": a,
                        "split_b": b,
                        "tau": float(tau) if np.isfinite(tau) else np.nan,
                        "top1_a": top_a[0],
                        "top1_b": top_b[0],
                        "top1_switch": bool(top_a[0] != top_b[0]),
                        "top3_overlap": int(len(set(top_a[:3]) & set(top_b[:3]))),
                        "top3_jaccard": float(len(set(top_a[:3]) & set(top_b[:3])) / len(set(top_a[:3]) | set(top_b[:3]))),
                    }
                )
    shift_df = pd.DataFrame(rows_shift)
    shift_df.to_csv(out_dir / "ranking_reversal_across_splits.csv", index=False)

    summary = {
        "mae_vs_nll_cases": int(len(metric_df)),
        "mae_vs_nll_top1_switch_rate": float(metric_df["top1_switch"].mean()) if len(metric_df) else None,
        "mae_vs_nll_tau_mean": float(metric_df["tau_mae_vs_nll"].mean()) if len(metric_df) else None,
        "split_pair_cases": int(len(shift_df)),
        "split_pair_top1_switch_rate": float(shift_df["top1_switch"].mean()) if len(shift_df) else None,
        "split_pair_tau_mean": float(shift_df["tau"].mean()) if len(shift_df) else None,
    }
    write_json(out_dir / "ranking_reversal_manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chembl", default="data/processed/chembl37_canal20.parquet")
    parser.add_argument("--bindingdb", default="data/processed/bindingdb_articles_202608_canal20.parquet")
    parser.add_argument("--metrics", default="artifacts/combined/metrics_long.csv")
    parser.add_argument("--out", default="artifacts/audits")
    parser.add_argument("--overlap-only", action="store_true", help="Build the independent BindingDB cohorts and stop before metric audits.")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    overlap = overlap_audit(Path(args.chembl), Path(args.bindingdb), out_dir)
    if args.overlap_only:
        print(json.dumps({"overlap": overlap}, indent=2, sort_keys=True))
        return
    accounting = run_accounting(Path(args.metrics), out_dir)
    ranking = ranking_reversal(Path(args.metrics), out_dir)
    summary = {
        "overlap": overlap,
        "run_accounting": accounting,
        "ranking_reversal": ranking,
    }
    write_json(out_dir / "audit_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
