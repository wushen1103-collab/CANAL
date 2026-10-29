#!/usr/bin/env python
"""Add a label-free source-context tail split to a frozen ChEMBL panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"}).astype(float)


def present(series: pd.Series) -> pd.Series:
    values = series.astype("string")
    return (values.notna() & values.ne("")).astype(float)


def source_features(frame: pd.DataFrame) -> pd.DataFrame:
    assay_type = frame["assay_type"].astype("string").fillna("MISSING")
    organism = frame["assay_organism"].astype("string").fillna("MISSING")
    organism_class = pd.Series(
        np.where(organism.eq("Homo sapiens"), "human", np.where(organism.eq("MISSING"), "missing", "other")),
        index=frame.index,
    )
    feature_frame = pd.DataFrame(index=frame.index)
    for value in ["A", "B", "F", "T", "MISSING"]:
        feature_frame[f"assay_type_{value}"] = assay_type.eq(value).astype(float)
    for value in ["human", "missing", "other"]:
        feature_frame[f"organism_{value}"] = organism_class.eq(value).astype(float)
    feature_frame["confidence_8"] = pd.to_numeric(frame["confidence_score"], errors="coerce").eq(8).astype(float)
    feature_frame["potential_duplicate"] = truthy(frame["potential_duplicate"])
    feature_frame["validity_comment_present"] = present(frame["data_validity_comment"])
    feature_frame["activity_comment_present"] = present(frame["activity_comment"])
    return feature_frame


def assign_by_rows(group_table: pd.DataFrame) -> dict[str, str]:
    ordered = group_table.sort_values(["source_tail_score", "source_group"], ascending=[False, True])
    total_rows = int(ordered["rows"].sum())
    assignment = {str(group): "train" for group in ordered["source_group"]}
    targets = {"test": TEST_FRACTION * total_rows, "val": VALIDATION_FRACTION * total_rows}
    maxima = {"test": 0.30 * total_rows, "val": 0.20 * total_rows}
    selected = {"test": 0, "val": 0}
    for row in ordered.itertuples(index=False):
        group = str(row.source_group)
        rows = int(row.rows)
        candidates = [
            split
            for split in ["test", "val"]
            if selected[split] < targets[split] and selected[split] + rows <= maxima[split]
        ]
        if not candidates:
            continue
        split = min(candidates, key=lambda value: selected[value] / targets[value])
        assignment[group] = split
        selected[split] += rows

    for split in ["test", "val"]:
        if selected[split] > 0:
            continue
        candidates = ordered.loc[ordered["source_group"].astype(str).map(assignment).eq("train")]
        if len(candidates):
            row = candidates.sort_values(["rows", "source_tail_score"], ascending=[True, False]).iloc[0]
            assignment[str(row["source_group"])] = split
            selected[split] += int(row["rows"])

    # Relation labels are used only as a minimum evaluability constraint, never
    # in the source-tail score or ordering.
    for relation_column in ["exact_rows", "censored_rows"]:
        test_groups = ordered["source_group"].astype(str).map(assignment).eq("test")
        if ordered.loc[test_groups, relation_column].sum() > 0:
            continue
        candidates = ordered.loc[
            ordered["source_group"].astype(str).map(assignment).eq("train")
            & ordered[relation_column].gt(0)
        ]
        if len(candidates):
            row = candidates.sort_values(["source_tail_score", "rows"], ascending=[False, True]).iloc[0]
            assignment[str(row["source_group"])] = "test"
    return assignment


def add_source_context_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy()
    output["split_s5"] = pd.NA
    diagnostics = []
    for target, target_frame in output.groupby("target_chembl_id", sort=True):
        features = source_features(target_frame)
        source_groups = target_frame["document_chembl_id"].astype(str)
        features["source_group"] = source_groups
        group_features = features.groupby("source_group", sort=True).mean()
        group_features["rows"] = target_frame.groupby(source_groups).size()
        relation = target_frame["p_relation"]
        group_features["exact_rows"] = relation.eq("=").groupby(
            source_groups
        ).sum()
        group_features["censored_rows"] = relation.ne("=").groupby(
            source_groups
        ).sum()
        context_columns = [
            column for column in group_features.columns if column not in {"rows", "exact_rows", "censored_rows"}
        ]
        center = group_features[context_columns].median(axis=0)
        scale = group_features[context_columns].std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
        standardized = (group_features[context_columns] - center) / scale
        group_features["source_tail_score"] = np.sqrt((standardized**2).mean(axis=1))
        group_table = group_features.reset_index()
        assignment = assign_by_rows(group_table)
        index = target_frame.index
        output.loc[index, "split_s5"] = source_groups.map(assignment).to_numpy()
        test_parents = set(
            output.loc[index]
            .loc[output.loc[index, "split_s5"].eq("test"), "parent_inchikey"]
            .dropna()
            .astype(str)
        )
        parent_values = output.loc[index, "parent_inchikey"].astype("string")
        test_overlap = output.loc[index, "split_s5"].ne("test") & parent_values.isin(test_parents)
        output.loc[index[test_overlap.to_numpy()], "split_s5"] = "exclude"
        validation_parents = set(
            output.loc[index]
            .loc[output.loc[index, "split_s5"].eq("val"), "parent_inchikey"]
            .dropna()
            .astype(str)
        )
        validation_overlap = output.loc[index, "split_s5"].eq("train") & parent_values.isin(validation_parents)
        output.loc[index[validation_overlap.to_numpy()], "split_s5"] = "exclude"

        # Close transitive parent/document paths, not only direct parent overlap.
        # Keep the covariate-selected test/validation groups fixed and exclude
        # rows from earlier partitions when their connected component crosses a
        # protected boundary.
        test_components = set(
            output.loc[index]
            .loc[output.loc[index, "split_s5"].eq("test"), "leakage_group"]
            .dropna()
        )
        component_test_overlap = (
            output.loc[index, "split_s5"].ne("test")
            & output.loc[index, "leakage_group"].isin(test_components)
        )
        output.loc[index[component_test_overlap.to_numpy()], "split_s5"] = "exclude"
        validation_components = set(
            output.loc[index]
            .loc[output.loc[index, "split_s5"].eq("val"), "leakage_group"]
            .dropna()
        )
        component_validation_overlap = (
            output.loc[index, "split_s5"].eq("train")
            & output.loc[index, "leakage_group"].isin(validation_components)
        )
        output.loc[index[component_validation_overlap.to_numpy()], "split_s5"] = "exclude"
        score_by_group = group_features["source_tail_score"].to_dict()
        for split, split_frame in output.loc[index].groupby("split_s5", sort=True):
            source = source_features(split_frame)
            row_scores = split_frame["document_chembl_id"].astype(str).map(score_by_group)
            diagnostics.append(
                {
                    "target": target,
                    "split": split,
                    "rows": len(split_frame),
                    "row_fraction": len(split_frame) / len(target_frame),
                    "groups": split_frame["document_chembl_id"].nunique(),
                    "exact_rows": int(split_frame["p_relation"].eq("=").sum()),
                    "censored_rows": int(split_frame["p_relation"].ne("=").sum()),
                    "source_tail_score_mean": float(row_scores.mean()),
                    "assay_type_A_or_F_or_T_fraction": float(
                        split_frame["assay_type"].isin(["A", "F", "T"]).mean()
                    ),
                    "nonhuman_or_missing_fraction": float(
                        (~split_frame["assay_organism"].eq("Homo sapiens").fillna(False)).mean()
                    ),
                    "confidence_8_fraction": float(
                        pd.to_numeric(split_frame["confidence_score"], errors="coerce").eq(8).mean()
                    ),
                    "potential_duplicate_fraction": float(source["potential_duplicate"].mean()),
                    "validity_comment_fraction": float(source["validity_comment_present"].mean()),
                    "activity_comment_fraction": float(source["activity_comment_present"].mean()),
                }
            )
    return output, pd.DataFrame(diagnostics)


def leakage_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, target_frame in frame.groupby("target_chembl_id", sort=True):
        row = {"target": target}
        partitions = {
            name: target_frame.loc[target_frame["split_s5"].eq(name)]
            for name in ("train", "val", "test")
        }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            for label, column in {
                "parent": "parent_inchikey",
                "document": "document_chembl_id",
                "assay": "assay_chembl_id",
                "leakage_group": "leakage_group",
            }.items():
                left_values = set(partitions[left][column].dropna().astype(str))
                right_values = set(partitions[right][column].dropna().astype(str))
                row[f"{label}_{left}_{right}_overlap"] = len(left_values & right_values)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/chembl35_ic50_top20_v2.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/chembl35_ic50_top20_v2_s5.parquet"),
    )
    parser.add_argument(
        "--report-prefix",
        type=Path,
        default=Path("reports/evidence_completion_2026-09-03/source_context_s5"),
    )
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    output, diagnostics = add_source_context_split(frame)
    if output["split_s5"].isna().any():
        raise RuntimeError("S5 assignment contains missing values")
    overlap = leakage_audit(output)
    overlap_columns = [column for column in overlap if column.endswith("_overlap")]
    if overlap[overlap_columns].to_numpy().sum() != 0:
        raise RuntimeError("S5 leakage audit failed")
    required = diagnostics.loc[diagnostics["split"].isin(["train", "test"])]
    per_target_split = required.pivot(index="target", columns="split", values=["exact_rows", "censored_rows"])
    if per_target_split.isna().any().any() or (per_target_split.to_numpy() == 0).any():
        raise RuntimeError("S5 requires exact and censored records in every target train/test split")
    if diagnostics.loc[diagnostics["split"].eq("val")].groupby("target")["rows"].sum().reindex(
        output["target_chembl_id"].unique(), fill_value=0
    ).eq(0).any():
        raise RuntimeError("S5 requires a nonempty validation split for every target")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report_prefix.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    diagnostics.to_csv(args.report_prefix.with_name(args.report_prefix.name + "_diagnostics.csv"), index=False)
    overlap.to_csv(args.report_prefix.with_name(args.report_prefix.name + "_leakage.csv"), index=False)
    manifest = {
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "test_fraction_target": TEST_FRACTION,
        "validation_fraction_target": VALIDATION_FRACTION,
        "assignment_unit": "within-target document_chembl_id",
        "selection": "descending standardized distance from median source-context vector",
        "source_tail_score_label_free": True,
        "feasibility_constraint": "test must contain at least one exact and one censored record",
        "leakage_control": (
            "non-test rows sharing a parent/document connected component with test, and train rows "
            "sharing a component with validation, are excluded; parent, document, assay, and full "
            "component overlap are audited across all partition pairs"
        ),
        "scope": "target-specific models; parent, document, and assay IDs are disjoint within every target",
        "source_features": [
            "assay type",
            "assay organism class",
            "confidence score 8",
            "potential duplicate",
            "validity comment presence",
            "activity comment presence",
        ],
    }
    args.report_prefix.with_name(args.report_prefix.name + "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
