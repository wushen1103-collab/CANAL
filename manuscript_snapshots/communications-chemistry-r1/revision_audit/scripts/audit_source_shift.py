#!/usr/bin/env python
"""Quantify source and temporal shift severity for the frozen ChEMBL panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


SPLITS = ["S0", "S1", "S2", "S3", "S4"]
CATEGORICAL_COLUMNS = ["assay_type", "assay_organism", "confidence_score"]


def clean_category(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>").replace("", "<MISSING>")


def unseen_fraction(train: pd.Series, test: pd.Series) -> float:
    if len(train) == 0 or len(test) == 0:
        return np.nan
    train_values = set(clean_category(train))
    values = clean_category(test)
    return float((~values.isin(train_values)).mean()) if len(values) else np.nan


def js_divergence(train: pd.Series, test: pd.Series) -> float:
    if len(train) == 0 or len(test) == 0:
        return np.nan
    train = clean_category(train)
    test = clean_category(test)
    categories = sorted(set(train) | set(test))
    if not categories:
        return np.nan
    p = train.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(dtype=float)
    q = test.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(dtype=float)
    return float(jensenshannon(p, q, base=2.0) ** 2)


def set_overlap(train: pd.Series, test: pd.Series) -> int:
    left = set(clean_category(train)) - {"<MISSING>"}
    right = set(clean_category(test)) - {"<MISSING>"}
    return len(left & right)


def year_stats(frame: pd.DataFrame, prefix: str) -> dict[str, object]:
    years = pd.to_numeric(frame["document_year"], errors="coerce")
    return {
        f"{prefix}_year_n": int(years.notna().sum()),
        f"{prefix}_year_missing_fraction": float(years.isna().mean()) if len(years) else np.nan,
        f"{prefix}_year_min": years.min(),
        f"{prefix}_year_median": years.median(),
        f"{prefix}_year_max": years.max(),
    }


def audit_dataset(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(path)
    endpoint = str(frame["endpoint"].dropna().iloc[0])
    rows = []
    temporal_rows = []
    years = pd.to_numeric(frame["document_year"], errors="coerce")
    temporal_fallback = bool(years.notna().sum() < max(100, 0.2 * len(frame)))
    for target, target_frame in frame.groupby("target_chembl_id", sort=True):
        for split in SPLITS:
            split_col = f"split_{split.lower()}"
            train = target_frame.loc[target_frame[split_col].eq("train")]
            validation = target_frame.loc[target_frame[split_col].eq("val")]
            test = target_frame.loc[target_frame[split_col].eq("test")]
            row: dict[str, object] = {
                "dataset": path.stem,
                "endpoint": endpoint,
                "target": target,
                "split": split,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "train_groups": train["leakage_group"].nunique() if "leakage_group" in frame else np.nan,
                "test_groups": test["leakage_group"].nunique() if "leakage_group" in frame else np.nan,
                "assay_overlap": set_overlap(train["assay_chembl_id"], test["assay_chembl_id"]),
                "document_overlap": set_overlap(train["document_chembl_id"], test["document_chembl_id"]),
                "parent_overlap": set_overlap(train["parent_inchikey"], test["parent_inchikey"]),
                "scaffold_overlap": set_overlap(train["scaffold"], test["scaffold"]),
            }
            for column in CATEGORICAL_COLUMNS:
                row[f"unseen_{column}_row_fraction"] = unseen_fraction(train[column], test[column])
                row[f"{column}_js_divergence"] = js_divergence(train[column], test[column])
            row.update(year_stats(train, "train"))
            row.update(year_stats(test, "test"))
            row["test_minus_train_median_year"] = (
                row["test_year_median"] - row["train_year_median"]
                if pd.notna(row["test_year_median"]) and pd.notna(row["train_year_median"])
                else np.nan
            )
            rows.append(row)
            if split == "S3":
                temporal = dict(row)
                temporal.update(year_stats(validation, "validation"))
                temporal["endpoint_level_temporal_fallback"] = temporal_fallback
                temporal["test_median_later_than_train"] = bool(
                    pd.notna(row["test_minus_train_median_year"])
                    and row["test_minus_train_median_year"] > 0
                )
                temporal_rows.append(temporal)
    return pd.DataFrame(rows), pd.DataFrame(temporal_rows)


def summarize(audit: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "unseen_assay_type_row_fraction",
        "unseen_assay_organism_row_fraction",
        "confidence_score_js_divergence",
        "assay_type_js_divergence",
        "assay_organism_js_divergence",
        "test_minus_train_median_year",
        "assay_overlap",
        "document_overlap",
        "parent_overlap",
        "scaffold_overlap",
    ]
    summary = (
        audit.groupby(["endpoint", "split"], sort=True)[metrics]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evidence_completion_2026-09-03/source_shift"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted((repo / "data" / "processed").glob("chembl35_*_top20_v2.parquet"))
    if len(paths) != 4:
        raise RuntimeError(f"Expected four frozen endpoint panels, observed {len(paths)}")
    audits = []
    temporal = []
    for path in paths:
        audit_frame, temporal_frame = audit_dataset(path)
        audits.append(audit_frame)
        temporal.append(temporal_frame)
    audit = pd.concat(audits, ignore_index=True)
    temporal_audit = pd.concat(temporal, ignore_index=True)
    summary = summarize(audit)
    audit.to_csv(output.with_name(output.name + "_by_target.csv"), index=False)
    temporal_audit.to_csv(output.with_name(output.name + "_temporal_audit.csv"), index=False)
    summary.to_csv(output.with_name(output.name + "_summary.csv"), index=False)
    manifest = {
        "datasets": [str(path) for path in paths],
        "splits": SPLITS,
        "categorical_shift": CATEGORICAL_COLUMNS,
        "js_divergence_base": 2,
        "temporal_fallback_rule": "endpoint-level dated rows < max(100, 0.2 * endpoint rows)",
    }
    output.with_name(output.name + "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(output.with_name(output.name + "_summary.csv"))


if __name__ == "__main__":
    main()

