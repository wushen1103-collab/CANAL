#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_chembl_canal import curate, load_chembl, sha256_file


def cohort_summary(test: pd.DataFrame, old: pd.DataFrame) -> dict[str, object]:
    old_docs = set(old["document_chembl_id"].dropna().astype(str))
    old_assays = set(old["assay_chembl_id"].dropna().astype(str))
    old_parents = set(old["parent_inchikey"].dropna().astype(str))
    return {
        "rows": int(len(test)),
        "targets": int(test["target_chembl_id"].nunique()),
        "documents": int(test["document_chembl_id"].nunique()),
        "assays": int(test["assay_chembl_id"].nunique()),
        "parents": int(test["parent_inchikey"].nunique()),
        "exact_rows": int(test["p_relation"].eq("=").sum()),
        "censored_rows": int(test["p_relation"].ne("=").sum()),
        "old_document_overlap": int(test["document_chembl_id"].astype(str).isin(old_docs).sum()),
        "old_assay_overlap": int(test["assay_chembl_id"].astype(str).isin(old_assays).sum()),
        "old_parent_overlap": int(test["parent_inchikey"].astype(str).isin(old_parents).sum()),
        "relations": {str(k): int(v) for k, v in test["p_relation"].value_counts().items()},
        "rows_by_target": {
            str(k): int(v) for k, v in test.groupby("target_chembl_id").size().sort_values(ascending=False).items()
        },
    }


def make_panel(old: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    development = old[old["split_s3"].isin(["train", "val"])].copy()
    development["split_update"] = development["split_s3"]
    test = test.copy()
    test["split_update"] = "test"
    columns = list(dict.fromkeys(list(development.columns) + list(test.columns)))
    return pd.concat(
        [development.reindex(columns=columns), test.reindex(columns=columns)],
        ignore_index=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build untouched ChEMBL36 release-update cohorts for frozen CANAL evaluation."
    )
    parser.add_argument("--chembl35-panel", type=Path, required=True)
    parser.add_argument("--chembl36-db", type=Path, required=True)
    parser.add_argument("--out-prefix", default="chembl36_ic50_release_update")
    args = parser.parse_args()

    old = pd.read_parquet(args.chembl35_panel).reset_index(drop=True)
    old = old[old["endpoint"].eq("IC50")].copy()
    targets = sorted(old["target_chembl_id"].dropna().astype(str).unique())
    if len(targets) != 20:
        raise ValueError(f"Expected 20 frozen IC50 targets, found {len(targets)}")

    with sqlite3.connect(args.chembl36_db) as conn:
        raw36 = load_chembl(conn, ["IC50"], targets)
    curated36, curation_stats = curate(raw36)

    old_docs = set(old["document_chembl_id"].dropna().astype(str))
    old_parents = set(old["parent_inchikey"].dropna().astype(str))
    has_doc = curated36["document_chembl_id"].notna()
    release_update = curated36[
        has_doc & ~curated36["document_chembl_id"].astype(str).isin(old_docs)
    ].copy()
    novel_parent = release_update[
        ~release_update["parent_inchikey"].astype(str).isin(old_parents)
    ].copy()

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / f"{args.out_prefix}.parquet"
    strict_path = out_dir / f"{args.out_prefix}_novel_parent.parquet"
    make_panel(old, release_update).to_parquet(all_path, index=False)
    make_panel(old, novel_parent).to_parquet(strict_path, index=False)

    manifest = {
        "protocol": {
            "development_release": "ChEMBL35",
            "evaluation_release": "ChEMBL36",
            "endpoint": "IC50",
            "targets": targets,
            "training_rows": "ChEMBL35 split_s3=train",
            "validation_rows": "ChEMBL35 split_s3=val",
            "release_update_test": "ChEMBL36 documents absent from the frozen ChEMBL35 IC50 panel",
            "novel_parent_test": "release_update_test with parent InChIKey absent from the entire frozen ChEMBL35 IC50 panel",
            "selection_rule_uses_labels": False,
        },
        "inputs": {
            "chembl35_panel": str(args.chembl35_panel),
            "chembl35_panel_sha256": sha256_file(args.chembl35_panel),
            "chembl36_db": str(args.chembl36_db),
            "chembl36_db_sha256": sha256_file(args.chembl36_db),
        },
        "chembl36_curation": curation_stats,
        "development": {
            "train_rows": int(old["split_s3"].eq("train").sum()),
            "validation_rows": int(old["split_s3"].eq("val").sum()),
            "excluded_seen_test_rows": int(old["split_s3"].eq("test").sum()),
        },
        "release_update": cohort_summary(release_update, old),
        "novel_parent": cohort_summary(novel_parent, old),
        "outputs": {"release_update": str(all_path), "novel_parent": str(strict_path)},
    }
    manifest_path = out_dir / f"{args.out_prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    target_table = pd.concat(
        [
            release_update.groupby(["target_chembl_id", "p_relation"]).size().rename("rows").reset_index().assign(cohort="release_update"),
            novel_parent.groupby(["target_chembl_id", "p_relation"]).size().rename("rows").reset_index().assign(cohort="novel_parent"),
        ],
        ignore_index=True,
    )
    target_table.to_csv(out_dir / f"{args.out_prefix}_target_counts.csv", index=False)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
