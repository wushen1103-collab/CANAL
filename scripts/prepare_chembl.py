#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


RDLogger.DisableLog("rdApp.*")

RELATIONS = ("=", "<", ">", "<=", ">=")
ENDPOINTS = ("IC50", "Ki", "EC50", "Kd")


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def sha256_file(path: str | Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_and_scaffold(smiles: str) -> tuple[str | None, str | None]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    can = Chem.MolToSmiles(mol, canonical=True)
    try:
        scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        scaff = ""
    if not scaff:
        scaff = can
    return can, scaff


def load_group_stats(conn: sqlite3.Connection, year_min: int, year_max: int) -> pd.DataFrame:
    sql = f"""
    WITH single_targets AS (
      SELECT tc.tid AS tid
      FROM target_components tc
      JOIN component_sequences cs ON tc.component_id = cs.component_id
      GROUP BY tc.tid
      HAVING COUNT(DISTINCT cs.accession) = 1
    )
    SELECT
      UPPER(a.standard_type) AS endpoint,
      ass.tid AS tid,
      td.chembl_id AS target_chembl_id,
      td.pref_name AS target_name,
      COUNT(*) AS n,
      SUM(CASE WHEN a.standard_relation = '=' THEN 1 ELSE 0 END) AS exact_n,
      SUM(CASE WHEN a.standard_relation != '=' THEN 1 ELSE 0 END) AS censored_n,
      SUM(CASE WHEN a.standard_relation IN ('<','<=') THEN 1 ELSE 0 END) AS lower_bound_n,
      SUM(CASE WHEN a.standard_relation IN ('>','>=') THEN 1 ELSE 0 END) AS upper_bound_n,
      COUNT(DISTINCT a.molregno) AS molecule_n,
      COUNT(DISTINCT ass.assay_id) AS assay_n,
      COUNT(DISTINCT ass.doc_id) AS doc_n,
      MIN(CAST(d.year AS INTEGER)) AS year_min,
      MAX(CAST(d.year AS INTEGER)) AS year_max
    FROM activities a
    JOIN assays ass ON a.assay_id = ass.assay_id
    JOIN docs d ON ass.doc_id = d.doc_id
    JOIN target_dictionary td ON ass.tid = td.tid
    JOIN single_targets st ON ass.tid = st.tid
    JOIN compound_structures cst ON a.molregno = cst.molregno
    WHERE UPPER(a.standard_type) IN ({",".join("?" for _ in ENDPOINTS)})
      AND a.standard_relation IN ({",".join("?" for _ in RELATIONS)})
      AND LOWER(a.standard_units) IN ('nm', 'nanomolar', 'nanomol/l')
      AND CAST(a.standard_value AS REAL) > 0
      AND a.standard_flag = 1
      AND td.target_type = 'SINGLE PROTEIN'
      AND ass.confidence_score >= 8
      AND CAST(d.year AS INTEGER) BETWEEN ? AND ?
      AND cst.canonical_smiles IS NOT NULL
    GROUP BY UPPER(a.standard_type), ass.tid, td.chembl_id, td.pref_name
    """
    params = list(ENDPOINTS) + list(RELATIONS) + [year_min, year_max]
    df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        raise RuntimeError("No target-endpoint group passed the ChEMBL filter.")
    df["censor_rate"] = df["censored_n"] / df["n"].clip(lower=1)
    df["lower_rate"] = df["lower_bound_n"] / df["n"].clip(lower=1)
    df["upper_rate"] = df["upper_bound_n"] / df["n"].clip(lower=1)
    return df


def select_groups(stats: pd.DataFrame, target_cap: int, min_rows: int, min_censor_rate: float, max_censor_rate: float) -> pd.DataFrame:
    cand = stats[
        (stats["n"] >= min_rows)
        & (stats["exact_n"] >= max(200, min_rows // 4))
        & (stats["censored_n"] >= max(50, min_rows // 20))
        & (stats["censor_rate"].between(min_censor_rate, max_censor_rate))
    ].copy()
    if cand.empty:
        cand = stats[(stats["n"] >= min_rows) & (stats["censored_n"] > 0)].copy()
    if cand.empty:
        raise RuntimeError("No target-endpoint group has both exact and censored labels.")
    # Favor usable scale, censor balance, endpoint diversity, and source diversity.
    cand["balance"] = 1.0 - (cand["censor_rate"] - 0.25).abs()
    cand["score"] = np.log1p(cand["n"]) + cand["balance"] + np.log1p(cand["assay_n"]) * 0.25 + np.log1p(cand["doc_n"]) * 0.15
    chosen = []
    used_tids = set()
    # First pass: avoid one very large target dominating the panel.
    for _, row in cand.sort_values("score", ascending=False).iterrows():
        key = (row["endpoint"], int(row["tid"]))
        if int(row["tid"]) in used_tids:
            continue
        chosen.append(key)
        used_tids.add(int(row["tid"]))
        if len(chosen) >= target_cap:
            break
    # Second pass: if needed, allow multiple endpoints per target.
    if len(chosen) < target_cap:
        present = set(chosen)
        for _, row in cand.sort_values("score", ascending=False).iterrows():
            key = (row["endpoint"], int(row["tid"]))
            if key in present:
                continue
            chosen.append(key)
            present.add(key)
            if len(chosen) >= target_cap:
                break
    out = pd.DataFrame(chosen, columns=["endpoint", "tid"])
    return out.merge(stats, on=["endpoint", "tid"], how="left")


def load_rows_for_groups(conn: sqlite3.Connection, groups: pd.DataFrame, year_min: int, year_max: int) -> pd.DataFrame:
    clauses = " OR ".join(["(UPPER(a.standard_type)=? AND ass.tid=?)" for _ in range(len(groups))])
    sql = f"""
    WITH single_targets AS (
      SELECT
        tc.tid AS tid,
        MIN(cs.accession) AS uniprot_accession,
        MIN(cs.sequence) AS target_sequence
      FROM target_components tc
      JOIN component_sequences cs ON tc.component_id = cs.component_id
      GROUP BY tc.tid
      HAVING COUNT(DISTINCT cs.accession) = 1
    )
    SELECT
      a.activity_id,
      md.chembl_id AS molecule_chembl_id,
      cst.canonical_smiles,
      cst.standard_inchi_key,
      td.chembl_id AS target_chembl_id,
      td.pref_name AS target_name,
      st.uniprot_accession,
      st.target_sequence,
      UPPER(a.standard_type) AS endpoint,
      a.standard_value,
      a.standard_units,
      a.standard_relation,
      a.pchembl_value,
      a.activity_comment,
      a.data_validity_comment,
      a.potential_duplicate,
      a.src_id AS activity_src_id,
      ass.assay_id,
      ass.chembl_id AS assay_chembl_id,
      ass.assay_type,
      ass.assay_test_type,
      ass.assay_category,
      ass.assay_organism,
      ass.assay_tax_id,
      ass.description AS assay_description,
      ass.bao_format,
      ass.confidence_score,
      ass.src_id AS assay_src_id,
      d.doc_id,
      d.chembl_id AS document_chembl_id,
      d.doc_type,
      d.journal,
      d.pubmed_id,
      d.doi,
      CAST(d.year AS INTEGER) AS publication_year
    FROM activities a
    JOIN assays ass ON a.assay_id = ass.assay_id
    JOIN docs d ON ass.doc_id = d.doc_id
    JOIN target_dictionary td ON ass.tid = td.tid
    JOIN single_targets st ON ass.tid = st.tid
    JOIN molecule_dictionary md ON a.molregno = md.molregno
    JOIN compound_structures cst ON a.molregno = cst.molregno
    WHERE ({clauses})
      AND UPPER(a.standard_type) IN ({",".join("?" for _ in ENDPOINTS)})
      AND a.standard_relation IN ({",".join("?" for _ in RELATIONS)})
      AND LOWER(a.standard_units) IN ('nm', 'nanomolar', 'nanomol/l')
      AND CAST(a.standard_value AS REAL) > 0
      AND a.standard_flag = 1
      AND td.target_type = 'SINGLE PROTEIN'
      AND ass.confidence_score >= 8
      AND CAST(d.year AS INTEGER) BETWEEN ? AND ?
      AND cst.canonical_smiles IS NOT NULL
    """
    params: list[object] = []
    for _, row in groups.iterrows():
        params.extend([str(row["endpoint"]).upper(), int(row["tid"])])
    params.extend(list(ENDPOINTS) + list(RELATIONS) + [year_min, year_max])
    return pd.read_sql_query(sql, conn, params=params)


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["standard_value_nm"] = pd.to_numeric(out["standard_value"], errors="coerce")
    out = out[out["standard_value_nm"] > 0].copy()
    threshold = 9.0 - np.log10(out["standard_value_nm"].astype(float))
    pchembl = pd.to_numeric(out["pchembl_value"], errors="coerce")
    out["p_activity_threshold"] = threshold
    # Exact labels use pChEMBL when available; censored labels use the transformed threshold.
    out["y_observed"] = np.where(out["standard_relation"].eq("="), pchembl.fillna(threshold), threshold)
    out["is_censored"] = ~out["standard_relation"].eq("=")
    # In p-activity scale: raw '< nM' means latent activity is above the threshold;
    # raw '> nM' means latent activity is below the threshold.
    out["censor_type"] = "exact"
    out.loc[out["standard_relation"].isin(["<", "<="]), "censor_type"] = "lower"
    out.loc[out["standard_relation"].isin([">", ">="]), "censor_type"] = "upper"
    out["target_endpoint"] = out["target_chembl_id"].astype(str) + "::" + out["endpoint"].astype(str)
    out["publication_year"] = pd.to_numeric(out["publication_year"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_observed", "publication_year"])
    out["publication_year"] = out["publication_year"].astype(int)
    return out


def deterministic_downsample(df: pd.DataFrame, target_rows: int | None, max_per_group: int | None, seed: int) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    if max_per_group:
        parts = []
        for _, g in out.groupby("target_endpoint", sort=False):
            if len(g) > max_per_group:
                # Keep exact/censored mix within each target-endpoint.
                parts.append(
                    g.groupby("censor_type", group_keys=False)
                    .apply(lambda x: x.sample(n=max(1, int(round(max_per_group * len(x) / len(g)))), random_state=seed) if len(x) else x)
                    .head(max_per_group)
                )
            else:
                parts.append(g)
        out = pd.concat(parts, ignore_index=True)
    if target_rows and len(out) > target_rows:
        weights = out.groupby("target_endpoint")["activity_id"].transform("count")
        frac_by_group = target_rows / len(out)
        out = (
            out.groupby(["target_endpoint", "censor_type"], group_keys=False)
            .apply(lambda x: x.sample(n=max(1, min(len(x), int(round(len(x) * frac_by_group)))), random_state=seed))
            .sample(frac=1.0, random_state=seed)
        )
        if len(out) > target_rows:
            out = out.sample(n=target_rows, random_state=seed)
    return out.reset_index(drop=True)


def add_chemistry(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    unique = pd.Series(df["canonical_smiles"].dropna().unique())
    with Pool(processes=max(1, workers)) as pool:
        rows = list(tqdm(pool.imap(canonical_and_scaffold, unique.tolist(), chunksize=256), total=len(unique), desc="rdkit"))
    cmap = dict(zip(unique.tolist(), rows))
    can = []
    scaffolds = []
    for smi in tqdm(df["canonical_smiles"].tolist(), desc="map_chemistry"):
        c, s = cmap.get(smi, (None, None))
        can.append(c)
        scaffolds.append(s)
    out = df.copy()
    out["canonical_smiles_rdkit"] = can
    out["bemis_murcko_scaffold"] = scaffolds
    out = out.dropna(subset=["canonical_smiles_rdkit"]).copy()
    out["molecule_key"] = out["standard_inchi_key"].fillna(out["canonical_smiles_rdkit"])
    out["first_seen_molecule_year"] = out.groupby("molecule_key")["publication_year"].transform("min")
    out["first_seen_scaffold_year"] = out.groupby("bemis_murcko_scaffold")["publication_year"].transform("min")
    out["first_seen_target_year"] = out.groupby("target_chembl_id")["publication_year"].transform("min")
    out["is_first_seen_molecule"] = out["publication_year"].eq(out["first_seen_molecule_year"])
    out["is_first_seen_scaffold"] = out["publication_year"].eq(out["first_seen_scaffold_year"])
    out["is_first_seen_target"] = out["publication_year"].eq(out["first_seen_target_year"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--out", default="data/processed/chembl37_canal20.parquet")
    parser.add_argument("--year-min", type=int, default=2010)
    parser.add_argument("--year-max", type=int, default=2025)
    parser.add_argument("--target-cap", type=int, default=20)
    parser.add_argument("--min-rows", type=int, default=2500)
    parser.add_argument("--min-censor-rate", type=float, default=0.05)
    parser.add_argument("--max-censor-rate", type=float, default=0.75)
    parser.add_argument("--max-per-group", type=int, default=12000)
    parser.add_argument("--target-rows", type=int, default=168008)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--workers", type=int, default=max(1, min(96, (os.cpu_count() or 4) - 30)))
    args = parser.parse_args()

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    with sqlite3.connect(args.sqlite) as conn:
        stats = load_group_stats(conn, args.year_min, args.year_max)
        selected = select_groups(stats, args.target_cap, args.min_rows, args.min_censor_rate, args.max_censor_rate)
        selected.to_csv(out_path.with_suffix(".selected_groups.csv"), index=False)
        stats.sort_values("n", ascending=False).to_csv(out_path.with_suffix(".all_group_stats.csv"), index=False)
        rows = load_rows_for_groups(conn, selected, args.year_min, args.year_max)
    rows = normalize_labels(rows)
    rows = deterministic_downsample(rows, args.target_rows, args.max_per_group, args.seed)
    rows = add_chemistry(rows, workers=args.workers)
    rows = rows.sort_values(["target_endpoint", "publication_year", "activity_id"]).reset_index(drop=True)
    rows.to_parquet(out_path, index=False)
    manifest = {
        "sqlite": str(args.sqlite),
        "sqlite_sha256": sha256_file(args.sqlite),
        "out": str(out_path),
        "rows": int(len(rows)),
        "target_endpoints": int(rows["target_endpoint"].nunique()),
        "targets": int(rows["target_chembl_id"].nunique()),
        "endpoints": sorted(rows["endpoint"].dropna().unique().tolist()),
        "molecules": int(rows["molecule_key"].nunique()),
        "scaffolds": int(rows["bemis_murcko_scaffold"].nunique()),
        "assays": int(rows["assay_chembl_id"].nunique()),
        "docs": int(rows["document_chembl_id"].nunique()),
        "years": [int(rows["publication_year"].min()), int(rows["publication_year"].max())],
        "censor_rate": float(rows["is_censored"].mean()),
        "relation_counts": rows["standard_relation"].value_counts(dropna=False).to_dict(),
        "censor_type_counts": rows["censor_type"].value_counts(dropna=False).to_dict(),
        "seed": int(args.seed),
        "target_rows_requested": int(args.target_rows) if args.target_rows else None,
        "max_per_group": int(args.max_per_group) if args.max_per_group else None,
    }
    write_json(out_path.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
