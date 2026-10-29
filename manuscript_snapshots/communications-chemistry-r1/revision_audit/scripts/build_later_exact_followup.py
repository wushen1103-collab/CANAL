#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PAIR_KEY = ["target_chembl_id", "parent_inchikey"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a ChEMBL35 censored-only to ChEMBL36 later-exact follow-up panel. "
            "Only ChEMBL35 S3 training/validation rows are retained for model fitting."
        )
    )
    parser.add_argument("--chembl35", type=Path, required=True)
    parser.add_argument("--chembl36-update", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    old = pd.read_parquet(args.chembl35).reset_index(drop=True)
    update = pd.read_parquet(args.chembl36_update).reset_index(drop=True)
    old = old[old["endpoint"].eq("IC50")].copy()

    pair_status = (
        old.groupby(PAIR_KEY, as_index=False)
        .agg(
            chembl35_rows=("p_relation", "size"),
            chembl35_exact_rows=("p_relation", lambda x: int(x.eq("=").sum())),
            chembl35_censored_rows=("p_relation", lambda x: int(x.ne("=").sum())),
        )
    )
    training_censored = (
        old.loc[old["split_s3"].eq("train") & old["p_relation"].ne("=")]
        .groupby(PAIR_KEY)
        .size()
        .rename("chembl35_training_censored_rows")
        .reset_index()
    )
    pair_status = pair_status.merge(training_censored, on=PAIR_KEY, how="left")
    pair_status["chembl35_training_censored_rows"] = (
        pair_status["chembl35_training_censored_rows"].fillna(0).astype(int)
    )
    eligible_pairs = pair_status.loc[
        pair_status["chembl35_exact_rows"].eq(0)
        & pair_status["chembl35_censored_rows"].gt(0)
        & pair_status["chembl35_training_censored_rows"].gt(0)
    ].copy()

    later_exact = update.loc[
        update["split_update"].eq("test") & update["p_relation"].eq("=")
    ].merge(eligible_pairs, on=PAIR_KEY, how="inner", validate="many_to_one")
    development = update.loc[update["split_update"].isin(["train", "val"])].copy()
    development["split_s3"] = development["split_update"]
    later_exact["split_s3"] = "test"

    base_columns = list(update.columns)
    panel = pd.concat(
        [development.reindex(columns=base_columns), later_exact.reindex(columns=base_columns)],
        ignore_index=True,
    )
    panel = panel.sort_values(
        ["target_chembl_id", "split_s3", "parent_inchikey", "document_chembl_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.out, index=False)

    target_counts = (
        later_exact.groupby(["target_chembl_id", "target_pref_name"], dropna=False)
        .agg(later_exact_rows=("p_activity", "size"), pairs=("parent_inchikey", "nunique"))
        .reset_index()
        .sort_values(["pairs", "later_exact_rows"], ascending=False)
    )
    target_counts.to_csv(args.out.with_suffix(".target_counts.csv"), index=False)
    manifest = {
        "definition": (
            "Compound-target pairs with at least one inequality-qualified IC50 record in "
            "ChEMBL35 S3 training, no exact IC50 record anywhere in the frozen ChEMBL35 "
            "panel, and at least one exact IC50 record in a new ChEMBL36 document."
        ),
        "development_rows": int(len(development)),
        "later_exact_rows": int(len(later_exact)),
        "later_exact_pairs": int(later_exact[PAIR_KEY].drop_duplicates().shape[0]),
        "targets": int(later_exact["target_chembl_id"].nunique()),
        "target_counts": target_counts.to_dict(orient="records"),
        "chembl35": str(args.chembl35),
        "chembl36_update": str(args.chembl36_update),
        "output": str(args.out),
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
