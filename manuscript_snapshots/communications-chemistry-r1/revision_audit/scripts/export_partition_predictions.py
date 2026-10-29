#!/usr/bin/env python
"""Export validation and test predictions for reviewer-facing reanalyses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from canal.censored_lgbm import train_censored_lgbm
from canal.resources import apply_thread_env, cpu_threads
from run_problem_check import (
    calibrate_sigma,
    featurize,
    filter_requested_panel,
    train_model,
    with_metadata_features,
)


def method_name(treatment: str) -> str:
    return {
        "exact_only": "LGBM Exact-only",
        "censored_as_exact": "LGBM Censored-as-exact",
        "censored_likelihood": "CANAL",
    }[treatment]


def fit_and_export(
    df: pd.DataFrame,
    x_all,
    target: str,
    split: str,
    seed: int,
    treatment: str,
    n_jobs: int,
    n_estimators: int,
    metadata_mode: str,
    sigma_calibration: str,
    censor_weight: float,
    anchor_weight: float,
    right_censor_weight: float,
    right_anchor_weight: float,
) -> pd.DataFrame | None:
    split_col = f"split_{split.lower()}"
    sub = df.loc[df["target_chembl_id"].eq(target)].copy()
    train = sub.loc[sub[split_col].eq("train")].copy()
    val = sub.loc[sub[split_col].eq("val")].copy()
    test = sub.loc[sub[split_col].eq("test")].copy()
    if treatment == "exact_only":
        train = train.loc[train["p_relation"].eq("=")].copy()
    if len(train) < 100 or len(test) < 30:
        return None

    x_train = x_all[train.index.to_numpy()]
    x_val = x_all[val.index.to_numpy()]
    x_test = x_all[test.index.to_numpy()]
    x_train, x_val, x_test = with_metadata_features(
        train, val, test, x_train, x_val, x_test, metadata_mode
    )
    y_train = train["p_activity"].to_numpy(dtype=float)
    if treatment == "censored_likelihood":
        model = train_censored_lgbm(
            x_train,
            y_train,
            train["p_relation"].to_numpy(),
            seed=seed,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            censor_weight=censor_weight,
            censored_anchor_weight=anchor_weight,
            right_censor_weight=right_censor_weight,
            right_censored_anchor_weight=right_anchor_weight,
        )
    else:
        model = train_model("lgbm", x_train, y_train, seed, n_jobs, n_estimators)

    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    exact_val = val["p_relation"].eq("=")
    if exact_val.sum() >= 20:
        fallback = float(
            np.std(val.loc[exact_val, "p_activity"].to_numpy() - val_pred[exact_val.to_numpy()])
        )
    else:
        fallback = float(np.std(y_train - train_pred))
    sigma = max(float(calibrate_sigma(val, val_pred, fallback, sigma_calibration)), 0.05)

    columns = [
        "row_id",
        "molecule_chembl_id",
        "parent_inchikey",
        "target_chembl_id",
        "assay_chembl_id",
        "document_chembl_id",
        "endpoint",
        "p_relation",
        "p_activity",
        "confidence_score",
        "potential_duplicate",
        "data_validity_comment",
        "activity_comment",
        "document_year",
        "scaffold",
    ]
    chunks = []
    for partition, frame, pred in (
        ("val", val, val_pred),
        ("test", test, model.predict(x_test)),
    ):
        out = frame[columns].copy()
        out["partition"] = partition
        out["split"] = split
        out["seed"] = seed
        out["treatment"] = treatment
        out["method"] = method_name(treatment)
        out["feature_set"] = "ecfp_desc"
        out["metadata_mode"] = metadata_mode
        out["n_estimators"] = n_estimators
        out["sigma"] = sigma
        out["pred"] = np.asarray(pred, dtype=float)
        chunks.append(out)
    return pd.concat(chunks, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["S1", "S2", "S3", "S4"])
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--treatments",
        nargs="+",
        default=["exact_only", "censored_as_exact", "censored_likelihood"],
        choices=["exact_only", "censored_as_exact", "censored_likelihood"],
    )
    parser.add_argument("--n-jobs", type=int, default=24)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--metadata-mode", default="source_light", choices=["source_light", "none"])
    parser.add_argument("--sigma-calibration", default="mixed_grid", choices=["mixed_grid", "exact"])
    parser.add_argument("--censor-weight", type=float, default=0.25)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--right-censor-weight", type=float, default=0.1)
    parser.add_argument("--right-anchor-weight", type=float, default=0.0)
    args = parser.parse_args()

    args.n_jobs = min(args.n_jobs, cpu_threads())
    apply_thread_env(args.n_jobs)
    raw = pd.read_parquet(args.dataset).reset_index(drop=True)
    raw["row_id"] = raw.index
    df, targets = filter_requested_panel(raw, ["IC50"], 20)
    targets = targets["IC50"]
    x_all = featurize(df["canonical_smiles_rdkit"], feature_set="ecfp_desc")
    chunks = []
    for target in targets:
        for split in args.splits:
            for seed in args.seeds:
                for treatment in args.treatments:
                    print(target, split, seed, treatment, flush=True)
                    chunk = fit_and_export(
                        df,
                        x_all,
                        target,
                        split,
                        seed,
                        treatment,
                        args.n_jobs,
                        args.n_estimators,
                        args.metadata_mode,
                        args.sigma_calibration,
                        args.censor_weight,
                        args.anchor_weight,
                        args.right_censor_weight,
                        args.right_anchor_weight,
                    )
                    if chunk is not None:
                        chunks.append(chunk)
    result = pd.concat(chunks, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.out, index=False)
    manifest = {
        "dataset": str(args.dataset),
        "rows": int(len(result)),
        "partitions": sorted(result["partition"].unique().tolist()),
        "splits": args.splits,
        "seeds": args.seeds,
        "treatments": args.treatments,
        "n_jobs": args.n_jobs,
        "n_estimators": args.n_estimators,
        "metadata_mode": args.metadata_mode,
        "sigma_calibration": args.sigma_calibration,
        "weights": {
            "left_likelihood": args.censor_weight,
            "left_anchor": args.anchor_weight,
            "right_likelihood": args.right_censor_weight,
            "right_anchor": args.right_anchor_weight,
        },
    }
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out} rows={len(result)}")


if __name__ == "__main__":
    main()
