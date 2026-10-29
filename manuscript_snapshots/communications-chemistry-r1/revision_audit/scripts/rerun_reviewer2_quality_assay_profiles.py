#!/usr/bin/env python3
"""Rerun only cells whose source/no-source worst-assay IDs differ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import run_problem_check as rpc


KEYS = ["endpoint", "target", "split", "seed", "treatment"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--quality-filter",
        required=True,
        choices=[
            "no_potential_duplicate",
            "no_validity_comment",
            "no_duplicate_or_validity_comment",
            "no_outside_typical_range",
            "no_potential_transcription_error",
        ],
    )
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    root = args.root.resolve()
    rpc.apply_thread_env(args.n_jobs)
    dataset = (
        root
        / "data"
        / "processed"
        / "reviewer2_quality"
        / f"chembl35_ic50_top20_v2_{args.quality_filter}.parquet"
    )
    mismatch_path = (
        root
        / "reports"
        / "reviewer2_validation_2026-10-28"
        / "quality_ablation"
        / "r2_9_worst_assay_mismatched_id_cells.csv"
    )
    output_dir = mismatch_path.parent / "assay_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output = output_dir / f"{args.quality_filter}_rerun_rows.csv"
    detail_output = output_dir / f"{args.quality_filter}_cross_model_assay_details.csv"

    cells = pd.read_csv(mismatch_path)
    cells = cells.loc[cells["quality_filter"].eq(args.quality_filter), KEYS].drop_duplicates()
    frame = pd.read_parquet(dataset).reset_index(drop=True)
    features = rpc.featurize(frame["canonical_smiles_rdkit"], feature_set="ecfp_desc")

    completed: set[tuple[object, ...]] = set()
    rows: list[dict[str, object]] = []
    if raw_output.exists():
        prior = pd.read_csv(raw_output)
        rows = prior.to_dict("records")
        completed = {
            tuple(record[key] for key in KEYS) + (record["profile_role"],)
            for record in rows
            if record.get("status") == "ok"
        }

    for cell in cells.itertuples(index=False):
        cell_values = {key: getattr(cell, key) for key in KEYS}
        for role, metadata_mode in (("source", "source_light"), ("nosource", "none")):
            completion_key = tuple(cell_values[key] for key in KEYS) + (role,)
            if completion_key in completed:
                continue
            print(
                f"[run] filter={args.quality_filter} target={cell.target} "
                f"split={cell.split} seed={cell.seed} role={role}",
                flush=True,
            )
            result = rpc.run_one(
                frame,
                features,
                endpoint=str(cell.endpoint),
                target=str(cell.target),
                split=str(cell.split),
                seed=int(cell.seed),
                treatment=str(cell.treatment),
                model_name="lgbm",
                n_jobs=args.n_jobs,
                n_estimators=300,
                feature_set="ecfp_desc",
                censor_weight=0.25,
                censored_anchor_weight=1.0,
                right_censor_weight=0.1,
                right_censored_anchor_weight=0.0,
                metadata_mode=metadata_mode,
                sigma_calibration="mixed_grid",
                posthoc_bias="none",
                posthoc_lambda=20.0,
            )
            if result is None:
                raise RuntimeError(f"No result for {completion_key}")
            result["quality_filter"] = args.quality_filter
            result["profile_role"] = role
            rows.append(result)
            pd.DataFrame(rows).to_csv(raw_output, index=False)

    reruns = pd.DataFrame(rows)
    expected_rows = 2 * len(cells)
    if len(reruns) != expected_rows or not reruns["status"].eq("ok").all():
        raise RuntimeError(
            f"Incomplete rerun: expected {expected_rows} successful rows, found {len(reruns)}"
        )

    detail_rows: list[dict[str, object]] = []
    for _, cell_rows in reruns.groupby(KEYS, sort=False):
        by_role = cell_rows.set_index("profile_role")
        source_profile = json.loads(by_role.loc["source", "worst_assay_profile_json"])
        nosource_profile = json.loads(by_role.loc["nosource", "worst_assay_profile_json"])
        source_worst = str(by_role.loc["source", "worst_assay_id"])
        nosource_worst = str(by_role.loc["nosource", "worst_assay_id"])
        for assay_id in sorted({source_worst, nosource_worst}):
            source_entry = source_profile[assay_id]
            nosource_entry = nosource_profile[assay_id]
            row = {key: by_role.iloc[0][key] for key in KEYS}
            row.update(
                {
                    "quality_filter": args.quality_filter,
                    "assay_id": assay_id,
                    "is_source_worst": assay_id == source_worst,
                    "is_nosource_worst": assay_id == nosource_worst,
                    "exact_test_n": int(source_entry["exact_n"]),
                    "source_mae": float(source_entry["mae"]),
                    "nosource_mae": float(nosource_entry["mae"]),
                    "source_minus_nosource": float(source_entry["mae"])
                    - float(nosource_entry["mae"]),
                }
            )
            detail_rows.append(row)
    details = pd.DataFrame(detail_rows)
    details.to_csv(detail_output, index=False)
    print(
        json.dumps(
            {
                "quality_filter": args.quality_filter,
                "mismatched_cells": int(len(cells)),
                "model_fits": int(len(reruns)),
                "detail_rows": int(len(details)),
                "minimum_exact_test_n": int(details["exact_test_n"].min()),
                "output": str(detail_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
