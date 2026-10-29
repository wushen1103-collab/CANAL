#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from canal.censored_lgbm import train_censored_lgbm
from canal.censored_xgb import train_censored_xgb
from canal.metrics import censored_gaussian_nll, gaussian_nll
from canal.resources import apply_thread_env, cpu_threads
from run_problem_check import (
    calibrate_sigma,
    featurize,
    filter_requested_panel,
    train_model,
    with_metadata_features,
)


def censored_row_nll(threshold: np.ndarray, relation: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    threshold = np.asarray(threshold, dtype=float)
    relation = np.asarray(relation).astype(str)
    mu = np.asarray(mu, dtype=float)
    sigma = max(float(sigma), 1e-4)
    z = (threshold - mu) / sigma
    cdf = np.clip(ndtr(z), 1e-12, 1.0 - 1e-12)
    probs = np.where(relation == ">", 1.0 - cdf, cdf)
    return -np.log(np.clip(probs, 1e-12, 1.0))


def censored_direction_ok(threshold: np.ndarray, relation: np.ndarray, mu: np.ndarray) -> np.ndarray:
    relation = np.asarray(relation).astype(str)
    return np.where(relation == ">", mu > threshold, mu < threshold)


def method_name(
    model_name: str,
    treatment: str,
    censor_weight: float,
    censored_anchor_weight: float,
    right_censor_weight: float | None,
    right_censored_anchor_weight: float | None,
) -> str:
    prefix = model_name.upper()
    if treatment == "exact_only":
        return f"{prefix} Exact-only"
    if treatment == "censored_as_exact":
        return f"{prefix} Censored-as-exact"
    name = f"{prefix} CANAL cw={censor_weight:g} aw={censored_anchor_weight:g}"
    if right_censor_weight is not None or right_censored_anchor_weight is not None:
        rcw = censor_weight if right_censor_weight is None else right_censor_weight
        raw = censored_anchor_weight if right_censored_anchor_weight is None else right_censored_anchor_weight
        name += f" right_cw={rcw:g} right_aw={raw:g}"
    return name


def train_and_predict(
    df: pd.DataFrame,
    x_all,
    endpoint: str,
    target: str,
    split: str,
    seed: int,
    treatment: str,
    model_name: str,
    n_jobs: int,
    n_estimators: int,
    censor_weight: float,
    censored_anchor_weight: float,
    right_censor_weight: float | None,
    right_censored_anchor_weight: float | None,
    sigma_calibration: str,
    metadata_mode: str,
    feature_set: str,
    min_test_rows: int = 30,
) -> pd.DataFrame | None:
    split_col = f"split_{split.lower()}"
    sub = df[(df["endpoint"] == endpoint) & (df["target_chembl_id"] == target)].copy()
    if sub.empty or split_col not in sub:
        return None

    train = sub[sub[split_col] == "train"].copy()
    val = sub[sub[split_col] == "val"].copy()
    test = sub[sub[split_col] == "test"].copy()
    if treatment == "exact_only":
        train = train[train["p_relation"] == "="]
    elif treatment in {"censored_as_exact", "censored_likelihood"}:
        pass
    else:
        raise ValueError(treatment)
    if len(train) < 100 or len(test) < min_test_rows:
        return None

    x_train = x_all[train.index.to_numpy()]
    y_train = train["p_activity"].to_numpy(dtype=float)
    x_val = x_all[val.index.to_numpy()] if len(val) else None
    x_test = x_all[test.index.to_numpy()]
    x_train, x_val, x_test = with_metadata_features(train, val, test, x_train, x_val, x_test, metadata_mode)

    if treatment == "censored_likelihood":
        if model_name == "xgb":
            model = train_censored_xgb(
                x_train,
                y_train,
                train["p_relation"].to_numpy(),
                seed=seed,
                n_jobs=n_jobs,
                n_estimators=n_estimators,
                censor_weight=censor_weight,
                censored_anchor_weight=censored_anchor_weight,
                right_censor_weight=right_censor_weight,
                right_censored_anchor_weight=right_censored_anchor_weight,
            )
        elif model_name == "lgbm":
            model = train_censored_lgbm(
                x_train,
                y_train,
                train["p_relation"].to_numpy(),
                seed=seed,
                n_jobs=n_jobs,
                n_estimators=n_estimators,
                censor_weight=censor_weight,
                censored_anchor_weight=censored_anchor_weight,
                right_censor_weight=right_censor_weight,
                right_censored_anchor_weight=right_censored_anchor_weight,
            )
        else:
            raise ValueError("prediction export supports censored_likelihood only for model=xgb or model=lgbm")
    else:
        model = train_model(model_name, x_train, y_train, seed, n_jobs, n_estimators)

    train_pred = model.predict(x_train)
    if x_val is not None and len(val):
        val_pred = model.predict(x_val)
        exact_val = val["p_relation"].eq("=")
        if exact_val.sum() >= 20:
            sigma = float(np.std(val.loc[exact_val, "p_activity"].to_numpy() - val_pred[exact_val.to_numpy()]))
        else:
            sigma = float(np.std(y_train - train_pred))
        sigma = calibrate_sigma(val, val_pred, sigma, sigma_calibration)
    else:
        sigma = float(np.std(y_train - train_pred))
    sigma = max(float(sigma), 0.05)

    pred = model.predict(x_test)
    out_cols = [
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
        "assay_type",
        "assay_organism",
        "document_year",
        "scaffold",
    ]
    out = test[out_cols].copy()
    out["split"] = split
    out["seed"] = seed
    out["treatment"] = treatment
    out["model"] = model_name
    out["feature_set"] = feature_set
    out["metadata_mode"] = metadata_mode
    out["method"] = method_name(
        model_name,
        treatment,
        censor_weight,
        censored_anchor_weight,
        right_censor_weight,
        right_censored_anchor_weight,
    )
    if metadata_mode != "none":
        out["method"] = out["method"] + f" [{metadata_mode}]"
    out["pred"] = pred
    out["sigma"] = sigma
    out["is_exact"] = out["p_relation"].eq("=")
    out["abs_error"] = np.where(out["is_exact"], np.abs(out["pred"] - out["p_activity"]), np.nan)
    out["gaussian_nll"] = np.where(
        out["is_exact"],
        0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((out["p_activity"] - out["pred"]) / sigma) ** 2,
        np.nan,
    )
    cens = ~out["is_exact"]
    out["censored_nll"] = np.nan
    out["censored_dir_ok"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    if cens.any():
        out.loc[cens, "censored_nll"] = censored_row_nll(
            out.loc[cens, "p_activity"].to_numpy(),
            out.loc[cens, "p_relation"].to_numpy(),
            out.loc[cens, "pred"].to_numpy(),
            sigma,
        )
        out.loc[cens, "censored_dir_ok"] = pd.array(censored_direction_ok(
            out.loc[cens, "p_activity"].to_numpy(),
            out.loc[cens, "p_relation"].to_numpy(),
            out.loc[cens, "pred"].to_numpy(),
        ), dtype="boolean")
    out["combo_gaussian_nll"] = gaussian_nll(
        test.loc[test["p_relation"].eq("="), "p_activity"].to_numpy(),
        pred[test["p_relation"].eq("=").to_numpy()],
        sigma,
    )
    out["combo_censored_nll"] = censored_gaussian_nll(
        test.loc[~test["p_relation"].eq("="), "p_activity"].to_numpy(),
        test.loc[~test["p_relation"].eq("="), "p_relation"].to_numpy(),
        pred[~test["p_relation"].eq("=").to_numpy()],
        sigma,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["S1", "S2", "S3", "S4"])
    parser.add_argument("--endpoints", nargs="+", default=["IC50"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027])
    parser.add_argument("--treatments", nargs="+", default=["exact_only", "censored_as_exact", "censored_likelihood"])
    parser.add_argument("--max-targets", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=24)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--min-test-rows", type=int, default=30)
    parser.add_argument("--model", default="xgb", choices=["xgb", "lgbm"])
    parser.add_argument("--feature-set", default="ecfp", choices=["ecfp", "desc", "ecfp_desc"])
    parser.add_argument("--censor-weight", type=float, default=0.1)
    parser.add_argument("--censored-anchor-weight", type=float, default=1.0)
    parser.add_argument("--right-censor-weight", type=float, default=None)
    parser.add_argument("--right-censored-anchor-weight", type=float, default=None)
    parser.add_argument(
        "--metadata-mode",
        default="none",
        choices=["none", "onehot_assay_doc", "source_light", "source_context", "source_quality"],
    )
    parser.add_argument("--sigma-calibration", default="mixed_grid", choices=["exact", "mixed_grid"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.n_jobs = min(args.n_jobs, cpu_threads())
    apply_thread_env(args.n_jobs)
    raw_df = pd.read_parquet(args.dataset).reset_index(drop=True)
    raw_df["row_id"] = raw_df.index
    df, targets_by_endpoint = filter_requested_panel(raw_df, args.endpoints, args.max_targets)
    print(f"[resource] n_jobs={args.n_jobs}")
    print(
        "[dataset] selected "
        f"{len(df)} / {len(raw_df)} rows for endpoints={args.endpoints} max_targets={args.max_targets}"
    )
    print(f"[features] building {args.feature_set} once for {len(df)} rows")
    x_all = featurize(df["canonical_smiles_rdkit"], feature_set=args.feature_set)

    chunks = []
    for endpoint in args.endpoints:
        for target in targets_by_endpoint.get(endpoint, []):
            for split in args.splits:
                for seed in args.seeds:
                    for treatment in args.treatments:
                        print(f"[run] endpoint={endpoint} target={target} split={split} seed={seed} treatment={treatment}")
                        chunk = train_and_predict(
                            df,
                            x_all,
                            endpoint,
                            target,
                            split,
                            seed,
                            treatment,
                            args.model,
                            args.n_jobs,
                            args.n_estimators,
                            args.censor_weight,
                            args.censored_anchor_weight,
                            args.right_censor_weight,
                            args.right_censored_anchor_weight,
                            args.sigma_calibration,
                            args.metadata_mode,
                            args.feature_set,
                            args.min_test_rows,
                        )
                        if chunk is not None:
                            chunks.append(chunk)
    result = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.out, index=False)
    result.head(5000).to_csv(args.out.with_suffix(".head.csv"), index=False)
    print(f"[done] wrote {args.out} rows={len(result)}")


if __name__ == "__main__":
    main()
