#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TREATMENTS = ("exact_only", "censored_as_exact", "bounded_mse")


def present_flag(frame: pd.DataFrame, column: str) -> pd.Series:
    text = frame[column].fillna("").astype(str).str.strip().str.lower()
    return (text.ne("") & text.ne("none") & text.ne("nan")).astype(float)


def source_light_descriptors(df: pd.DataFrame, split: str) -> tuple[pd.DataFrame, dict[str, object]]:
    split_col = f"split_{split.lower()}"
    train = df[df[split_col].eq("train")]
    out = pd.DataFrame(index=df.index)
    categories: dict[str, list[str]] = {}
    for column, prefix in (("assay_type", "src_type"), ("assay_organism", "src_org")):
        train_values = train[column].fillna("NA").astype(str)
        categories[column] = sorted(train_values.unique().tolist())
        values = df[column].fillna("NA").astype(str)
        for index, category in enumerate(categories[column]):
            out[f"{prefix}_{index:02d}"] = values.eq(category).astype(float)

    year = pd.to_numeric(df["document_year"], errors="coerce")
    train_year = pd.to_numeric(train["document_year"], errors="coerce")
    year_center = float(train_year.median()) if train_year.notna().any() else 2010.0
    out["src_confidence"] = pd.to_numeric(df["confidence_score"], errors="coerce").fillna(-1.0) / 10.0
    out["src_duplicate"] = pd.to_numeric(df["potential_duplicate"], errors="coerce").fillna(0.0)
    out["src_year_decade"] = (year.fillna(year_center) - year_center) / 10.0
    out["src_year_missing"] = year.isna().astype(float)
    out["src_validity_present"] = present_flag(df, "data_validity_comment")
    out["src_activity_comment_present"] = present_flag(df, "activity_comment")
    metadata: dict[str, object] = {
        "columns": out.columns.tolist(),
        "categories": categories,
        "year_center": year_center,
        "fit_partition": "train",
        "unseen_category_policy": "all-zero one-hot block",
    }
    return out, metadata


def target_matrix(
    df: pd.DataFrame,
    targets: list[str],
    split: str,
    treatment: str,
    descriptors: pd.DataFrame,
) -> pd.DataFrame:
    split_col = f"split_{split.lower()}"
    out = pd.DataFrame(
        {
            "smiles": df["canonical_smiles_rdkit"].astype(str),
            "split": df[split_col].astype(str),
        }
    )
    out = pd.concat([out, descriptors.reset_index(drop=True)], axis=1)
    for target in targets:
        out[target] = np.nan

    for target, indices in df.groupby("target_chembl_id", sort=False).groups.items():
        if treatment == "exact_only":
            keep = df.index.isin(indices) & (
                df["p_relation"].eq("=") | df[split_col].eq("test")
            )
            out.loc[keep, target] = df.loc[keep, "p_activity"].to_numpy(dtype=float)
        elif treatment == "censored_as_exact":
            keep = df.index.isin(indices)
            out.loc[keep, target] = df.loc[keep, "p_activity"].to_numpy(dtype=float)
        elif treatment == "bounded_mse":
            keep = df.index.isin(indices)
            values = df.loc[keep, "p_activity"].map(lambda value: f"{float(value):.10g}")
            relations = df.loc[keep, "p_relation"].astype(str)
            out[target] = out[target].astype(object)
            out.loc[keep, target] = (relations + values).where(relations.ne("="), values)
        else:
            raise ValueError(treatment)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["S1", "S2", "S3", "S4"])
    parser.add_argument("--treatments", nargs="+", default=list(TREATMENTS))
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset).reset_index(drop=True)
    df = df[df["endpoint"].eq("IC50")].reset_index(drop=True)
    targets = sorted(df["target_chembl_id"].astype(str).unique().tolist())
    if len(targets) != 20:
        raise ValueError(f"Expected 20 targets, found {len(targets)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    descriptor_columns: dict[str, list[str]] = {}
    source_metadata: dict[str, dict[str, object]] = {}
    for split in args.splits:
        split_col = f"split_{split.lower()}"
        if split_col not in df:
            raise KeyError(split_col)
        descriptors, metadata = source_light_descriptors(df, split)
        descriptor_columns[split] = descriptors.columns.tolist()
        source_metadata[split] = metadata
        for partition in ("val", "test"):
            path = args.output_dir / f"{split.lower()}_{partition}_smiles.csv"
            mask = df[split_col].eq(partition)
            prediction_input = df.loc[mask, ["canonical_smiles_rdkit"]].rename(
                columns={"canonical_smiles_rdkit": "smiles"}
            ).reset_index(drop=True)
            prediction_input = pd.concat(
                [prediction_input, descriptors.loc[mask].reset_index(drop=True)], axis=1
            )
            prediction_input.to_csv(path, index=False)
            files[f"{split}_{partition}"] = str(path)
        for treatment in args.treatments:
            path = args.output_dir / f"{split.lower()}_{treatment}.csv"
            target_matrix(df, targets, split, treatment, descriptors).to_csv(path, index=False)
            files[f"{split}_{treatment}"] = str(path)
            print(path)

    manifest = {
        "dataset": str(args.dataset),
        "rows": int(len(df)),
        "targets": targets,
        "splits": args.splits,
        "treatments": args.treatments,
        "descriptor_columns": descriptor_columns,
        "source_light": source_metadata,
        "files": files,
        "semantics": {
            "exact_only": "censored train/validation targets are missing; test thresholds remain numeric for external evaluation",
            "censored_as_exact": "all thresholds are treated as numeric point labels",
            "bounded_mse": "pActivity inequality prefixes are passed to Chemprop BoundedMSE",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
