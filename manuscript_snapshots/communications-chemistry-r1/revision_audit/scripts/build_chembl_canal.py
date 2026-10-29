#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from canal.chem import canonicalize_smiles, nm_to_p_activity, parent_inchikey, scaffold_smiles
from canal.splits import add_protocol_splits


ENDPOINTS = ("Ki", "Kd", "IC50", "EC50")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def find_db(version: int) -> Path:
    candidates = list((ROOT / "data" / "raw").glob(f"**/chembl_{version}.db"))
    if not candidates:
        raise FileNotFoundError(
            f"No chembl_{version}.db under data/raw. Run scripts/download_chembl_v35.sh first."
        )
    return candidates[0]


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def optional_column(conn: sqlite3.Connection, table: str, column: str, alias: str | None = None) -> str:
    alias = alias or column
    if column in table_columns(conn, table):
        return f"{table}.{column} AS {alias}"
    return f"NULL AS {alias}"


def sql_quote(values: list[str]) -> str:
    return ",".join("'" + v.replace("'", "''") + "'" for v in values)


def select_top_targets(conn: sqlite3.Connection, endpoints: list[str], smoke_targets: int) -> list[str]:
    endpoint_sql = sql_quote(endpoints)
    query = f"""
        SELECT td.chembl_id AS target_chembl_id, COUNT(*) AS n
        FROM activities
        JOIN assays ON activities.assay_id = assays.assay_id
        JOIN target_dictionary td ON assays.tid = td.tid
        JOIN compound_structures cs ON activities.molregno = cs.molregno
        WHERE td.organism = 'Homo sapiens'
          AND td.target_type = 'SINGLE PROTEIN'
          AND activities.standard_type IN ({endpoint_sql})
          AND activities.standard_units = 'nM'
          AND activities.standard_value IS NOT NULL
          AND activities.standard_value > 0
          AND activities.standard_relation IS NOT NULL
          AND cs.canonical_smiles IS NOT NULL
        GROUP BY td.chembl_id
        ORDER BY n DESC
        LIMIT {int(smoke_targets)}
    """
    return [row[0] for row in conn.execute(query).fetchall()]


def load_chembl(conn: sqlite3.Connection, endpoints: list[str], target_ids: list[str] | None) -> pd.DataFrame:
    endpoint_sql = sql_quote(endpoints)
    target_filter = ""
    if target_ids:
        target_filter = f"AND td.chembl_id IN ({sql_quote(target_ids)})"

    optional = [
        optional_column(conn, "activities", "pchembl_value"),
        optional_column(conn, "activities", "data_validity_comment"),
        optional_column(conn, "activities", "potential_duplicate"),
        optional_column(conn, "activities", "activity_comment"),
        optional_column(conn, "assays", "confidence_score"),
        optional_column(conn, "assays", "assay_type"),
        optional_column(conn, "assays", "assay_organism"),
        optional_column(conn, "docs", "year", "document_year"),
    ]
    optional_sql = ",\n            ".join(optional)
    query = f"""
        SELECT
            md.chembl_id AS molecule_chembl_id,
            cs.canonical_smiles AS canonical_smiles,
            cs.standard_inchi_key AS standard_inchi_key,
            td.chembl_id AS target_chembl_id,
            td.pref_name AS target_pref_name,
            td.target_type AS target_type,
            td.organism AS target_organism,
            assays.chembl_id AS assay_chembl_id,
            docs.chembl_id AS document_chembl_id,
            activities.standard_type AS endpoint,
            activities.standard_relation AS standard_relation,
            activities.standard_value AS standard_value,
            activities.standard_units AS standard_units,
            {optional_sql}
        FROM activities
        JOIN assays ON activities.assay_id = assays.assay_id
        LEFT JOIN docs ON assays.doc_id = docs.doc_id
        JOIN molecule_dictionary md ON activities.molregno = md.molregno
        JOIN compound_structures cs ON activities.molregno = cs.molregno
        JOIN target_dictionary td ON assays.tid = td.tid
        WHERE td.organism = 'Homo sapiens'
          AND td.target_type = 'SINGLE PROTEIN'
          AND activities.standard_type IN ({endpoint_sql})
          AND activities.standard_units = 'nM'
          AND activities.standard_value IS NOT NULL
          AND activities.standard_value > 0
          AND activities.standard_relation IS NOT NULL
          AND cs.canonical_smiles IS NOT NULL
          {target_filter}
    """
    return pd.read_sql_query(query, conn)


def curate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    stats: dict[str, int] = {"raw_rows": len(df)}
    values = [nm_to_p_activity(v, r) for v, r in zip(df["standard_value"], df["standard_relation"])]
    df = df.copy()
    df["p_activity"] = [v.p_activity if v else None for v in values]
    df["p_relation"] = [v.p_relation if v else None for v in values]
    stats["bad_relation_or_value"] = int(df["p_activity"].isna().sum())
    df = df.dropna(subset=["p_activity", "p_relation", "canonical_smiles", "standard_inchi_key"]).copy()

    tqdm.pandas(desc="canonicalize")
    df["canonical_smiles_rdkit"] = df["canonical_smiles"].progress_apply(canonicalize_smiles)
    stats["rdkit_standardization_failed"] = int(df["canonical_smiles_rdkit"].isna().sum())
    df = df.dropna(subset=["canonical_smiles_rdkit"]).copy()

    tqdm.pandas(desc="scaffold")
    df["scaffold"] = df["canonical_smiles_rdkit"].progress_apply(scaffold_smiles)
    df["parent_inchikey"] = df["standard_inchi_key"].apply(parent_inchikey)
    df["document_year"] = pd.to_numeric(df["document_year"], errors="coerce")
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    df["is_exact"] = df["p_relation"].eq("=")
    stats["curated_rows"] = len(df)
    stats["exact_rows"] = int(df["is_exact"].sum())
    stats["censored_rows"] = int((~df["is_exact"]).sum())
    return df, stats


def write_manifest(
    df: pd.DataFrame,
    reports: list,
    stats: dict,
    db_path: Path,
    out_prefix: str,
    target_ids: list[str] | None,
) -> None:
    manifest = {
        "out_prefix": out_prefix,
        "chembl_db_path": str(db_path),
        "chembl_db_sha256": sha256_file(db_path),
        "target_ids": target_ids,
        "counts": stats,
        "endpoint_counts": df["endpoint"].value_counts().to_dict(),
        "relation_counts": df["p_relation"].value_counts().to_dict(),
        "n_targets": int(df["target_chembl_id"].nunique()),
        "n_assays": int(df["assay_chembl_id"].nunique()),
        "n_documents": int(df["document_chembl_id"].nunique()),
        "n_scaffolds": int(df["scaffold"].nunique()),
        "split_reports": [r.__dict__ for r in reports],
    }
    path = ROOT / "data" / "processed" / f"{out_prefix}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    rows = []
    rows.append({"section": "counts", "name": "raw_rows", "value": stats["raw_rows"]})
    rows.append({"section": "counts", "name": "curated_rows", "value": stats["curated_rows"]})
    rows.append({"section": "counts", "name": "exact_rows", "value": stats["exact_rows"]})
    rows.append({"section": "counts", "name": "censored_rows", "value": stats["censored_rows"]})
    for endpoint, n in df["endpoint"].value_counts().items():
        rows.append({"section": "endpoint", "name": endpoint, "value": int(n)})
    for rel, n in df["p_relation"].value_counts().items():
        rows.append({"section": "relation", "name": rel, "value": int(n)})
    for r in reports:
        rows.extend(
            [
                {"section": r.split, "name": "train_rows", "value": r.train_rows},
                {"section": r.split, "name": "val_rows", "value": r.val_rows},
                {"section": r.split, "name": "test_rows", "value": r.test_rows},
                {"section": r.split, "name": "inchikey_train_test_overlap", "value": r.inchikey_train_test_overlap},
                {"section": r.split, "name": "document_train_test_overlap", "value": r.document_train_test_overlap},
            ]
        )
    pd.DataFrame(rows).to_csv(ROOT / "data" / "processed" / f"{out_prefix}_manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chembl-version", type=int, default=35)
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--out-prefix", default="chembl35_smoke")
    parser.add_argument("--endpoints", nargs="+", default=list(ENDPOINTS))
    parser.add_argument("--target-ids", nargs="*", default=None)
    parser.add_argument("--smoke-targets", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    db_path = args.db_path or find_db(args.chembl_version)
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        target_ids = args.target_ids
        if not target_ids and args.smoke_targets:
            target_ids = select_top_targets(conn, args.endpoints, args.smoke_targets)
            print(f"[targets] selected smoke targets: {target_ids}")
        raw = load_chembl(conn, args.endpoints, target_ids)

    curated, stats = curate(raw)
    with_splits, reports = add_protocol_splits(curated, seed=args.seed)
    out_path = out_dir / f"{args.out_prefix}.parquet"
    with_splits.to_parquet(out_path, index=False)
    write_manifest(with_splits, reports, stats, db_path, args.out_prefix, target_ids)
    print(f"[done] wrote {out_path} rows={len(with_splits)}")
    for report in reports:
        print(report)


if __name__ == "__main__":
    main()
