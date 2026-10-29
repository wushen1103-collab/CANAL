#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd


def overlap_with_table(
    conn: sqlite3.Connection, values: set[str], table: str, column: str
) -> set[str]:
    conn.execute("DROP TABLE IF EXISTS temp.wanted_ids")
    conn.execute("CREATE TEMP TABLE wanted_ids(value TEXT PRIMARY KEY)")
    conn.executemany("INSERT INTO wanted_ids VALUES (?)", [(value,) for value in values])
    query = f"SELECT source.{column} FROM {table} source JOIN wanted_ids wanted ON source.{column}=wanted.value"
    return {str(row[0]) for row in conn.execute(query)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-panel", type=Path, required=True)
    parser.add_argument("--chembl35-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    test = pd.read_parquet(
        args.update_panel,
        filters=[("split_update", "=", "test")],
        columns=["document_chembl_id", "assay_chembl_id", "parent_inchikey"],
    )
    documents = set(test["document_chembl_id"].dropna().astype(str))
    assays = set(test["assay_chembl_id"].dropna().astype(str))
    with sqlite3.connect(args.chembl35_db) as conn:
        schema_version = int(conn.execute("PRAGMA schema_version").fetchone()[0])
        document_table_rows = int(conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0])
        document_overlap = overlap_with_table(conn, documents, "docs", "chembl_id")
        assay_overlap = overlap_with_table(conn, assays, "assays", "chembl_id")

    audit = {
        "update_panel": str(args.update_panel),
        "chembl35_db": str(args.chembl35_db),
        "chembl35_schema_version": schema_version,
        "chembl35_document_table_rows": document_table_rows,
        "test_rows": int(len(test)),
        "test_documents": int(len(documents)),
        "test_assays": int(len(assays)),
        "chembl35_global_document_overlap": int(len(document_overlap)),
        "chembl35_global_assay_overlap": int(len(assay_overlap)),
        "overlapping_documents": sorted(document_overlap),
        "overlapping_assays": sorted(assay_overlap),
    }
    if document_overlap or assay_overlap:
        raise RuntimeError(json.dumps(audit, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
