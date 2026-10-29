#!/usr/bin/env python3
"""Record the exact information used to construct the S5 source-context split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/reviewer2_validation_2026-10-28/s5_protocol_audit"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/chembl35_ic50_top20_v2_s5.parquet"),
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=Path(
            "reports/evidence_completion_2026-09-03/source_context_s5_diagnostics.csv"
        ),
    )
    parser.add_argument(
        "--leakage",
        type=Path,
        default=Path(
            "reports/evidence_completion_2026-09-03/source_context_s5_leakage.csv"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    script = root / "scripts" / "build_source_context_shift.py"
    dataset = root / args.dataset
    diagnostics_path = root / args.diagnostics
    leakage_path = root / args.leakage
    diagnostics = pd.read_csv(diagnostics_path)
    leakage = pd.read_csv(leakage_path)
    frame = pd.read_parquet(dataset)

    overlap_columns = [column for column in leakage.columns if column.endswith("_overlap")]
    direct_overlap_columns = [
        column for column in overlap_columns if not column.startswith("leakage_group_")
    ]
    component_overlap_columns = [
        column for column in overlap_columns if column.startswith("leakage_group_")
    ]
    relation_feasibility = (
        diagnostics.loc[diagnostics["split"].isin(["train", "test"])]
        .pivot(index="target", columns="split", values=["exact_rows", "censored_rows"])
    )
    audit = {
        "recommended_description": (
            "response-value-free, full-panel covariate-informed, and relation-feasibility-constrained"
        ),
        "response_values_used_in_score_or_ordering": False,
        "p_activity_used_in_split_construction": False,
        "relation_labels_used": True,
        "relation_label_role": (
            "After covariate-based ordering, p_relation is used to ensure that the test partition "
            "contains at least one exact and one censored record per target."
        ),
        "full_panel_covariates_used": True,
        "full_panel_covariate_role": (
            "Per-document source-context means, medians, and standard deviations are computed "
            "before train/validation/test assignment, so eventual test covariates influence the split."
        ),
        "source_context_features": [
            "assay type",
            "assay organism class",
            "confidence score 8 indicator",
            "potential duplicate indicator",
            "validity-comment presence",
            "activity-comment presence",
        ],
        "assignment_unit": "document_chembl_id within target",
        "targets": int(frame["target_chembl_id"].nunique()),
        "rows": int(len(frame)),
        "all_targets_have_exact_and_censored_train_test": bool(
            relation_feasibility.notna().all().all()
            and (relation_feasibility.to_numpy() > 0).all()
        ),
        "zero_parent_document_assay_pairwise_overlap": bool(
            leakage[direct_overlap_columns].to_numpy().sum() == 0
        ),
        "zero_connected_component_pairwise_overlap": bool(
            leakage[component_overlap_columns].to_numpy().sum() == 0
        ),
        "all_pairwise_overlap_sum": int(leakage[overlap_columns].to_numpy().sum()),
        "interpretation": (
            "S5 is a designed stress test, not an untouched external distribution. It does not use "
            "numeric activity responses to rank documents, but it is not fully train-only and should "
            "not be described as completely label-free because relation labels enforce evaluability."
        ),
        "inputs": [
            {"path": str(script), "sha256": sha256(script)},
            {"path": str(dataset), "sha256": sha256(dataset)},
            {"path": str(diagnostics_path), "sha256": sha256(diagnostics_path)},
            {"path": str(leakage_path), "sha256": sha256(leakage_path)},
        ],
    }
    (output_dir / "r2_13_s5_protocol_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
