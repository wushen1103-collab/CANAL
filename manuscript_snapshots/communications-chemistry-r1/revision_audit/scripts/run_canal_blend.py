#!/usr/bin/env python
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
from canal.censored_xgb import train_censored_xgb
from canal.resources import apply_thread_env, cpu_threads
from run_problem_check import (
    calibrate_sigma,
    choose_targets,
    evaluate,
    featurize,
    mixed_validation_nll,
    train_model,
    with_metadata_features,
)


def exact_mae(frame: pd.DataFrame, pred: np.ndarray) -> float:
    exact = frame["p_relation"].eq("=")
    if not exact.any():
        return np.nan
    y = frame.loc[exact, "p_activity"].to_numpy(dtype=float)
    return float(np.mean(np.abs(pred[exact.to_numpy()] - y)))


def train_canal_model(
    model_name: str,
    x_train,
    y_train: np.ndarray,
    relation: np.ndarray,
    seed: int,
    n_jobs: int,
    n_estimators: int,
    censor_weight: float,
    censored_anchor_weight: float,
    right_censor_weight: float | None,
    right_censored_anchor_weight: float | None,
):
    if model_name == "xgb":
        return train_censored_xgb(
            x_train,
            y_train,
            relation,
            seed=seed,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            censor_weight=censor_weight,
            censored_anchor_weight=censored_anchor_weight,
            right_censor_weight=right_censor_weight,
            right_censored_anchor_weight=right_censored_anchor_weight,
        )
    if model_name == "lgbm":
        return train_censored_lgbm(
            x_train,
            y_train,
            relation,
            seed=seed,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            censor_weight=censor_weight,
            censored_anchor_weight=censored_anchor_weight,
            right_censor_weight=right_censor_weight,
            right_censored_anchor_weight=right_censored_anchor_weight,
        )
    raise ValueError("CANAL blend currently supports xgb/lgbm.")


def sigma_for(frame: pd.DataFrame, pred: np.ndarray, fallback: float, mode: str) -> float:
    exact = frame["p_relation"].eq("=")
    if exact.sum() >= 20:
        sigma = float(np.std(frame.loc[exact, "p_activity"].to_numpy(dtype=float) - pred[exact.to_numpy()]))
    else:
        sigma = fallback
    sigma = calibrate_sigma(frame, pred, max(sigma, 0.05), mode)
    return max(float(sigma), 0.05)


def run_one(
    df: pd.DataFrame,
    x_all,
    endpoint: str,
    target: str,
    split: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    split_col = f"split_{split.lower()}"
    sub = df[(df["endpoint"] == endpoint) & (df["target_chembl_id"] == target)].copy()
    if sub.empty or split_col not in sub:
        return None
    train_all = sub[sub[split_col] == "train"].copy()
    val = sub[sub[split_col] == "val"].copy()
    test = sub[sub[split_col] == "test"].copy()
    train_exact = train_all[train_all["p_relation"].eq("=")].copy()
    if len(train_exact) < 100 or len(train_all) < 100 or len(test) < 30:
        return {
            "endpoint": endpoint,
            "target": target,
            "split": split,
            "seed": seed,
            "method": "canal_blend",
            "model": args.model,
            "feature_set": args.feature_set,
            "metadata_mode": args.metadata_mode,
            "status": "too_few_rows",
            "train_rows": len(train_all),
            "train_exact_rows": len(train_exact),
            "val_rows": len(val),
            "test_rows": len(test),
        }

    x_exact = x_all[train_exact.index.to_numpy()]
    x_val_exact = x_all[val.index.to_numpy()] if len(val) else None
    x_test_exact = x_all[test.index.to_numpy()]
    x_exact, x_val_exact, x_test_exact = with_metadata_features(
        train_exact, val, test, x_exact, x_val_exact, x_test_exact, args.metadata_mode
    )

    x_train_all = x_all[train_all.index.to_numpy()]
    x_val_all = x_all[val.index.to_numpy()] if len(val) else None
    x_test_all = x_all[test.index.to_numpy()]
    x_train_all, x_val_all, x_test_all = with_metadata_features(
        train_all, val, test, x_train_all, x_val_all, x_test_all, args.metadata_mode
    )

    exact_model = train_model(
        args.model,
        x_exact,
        train_exact["p_activity"].to_numpy(dtype=float),
        seed,
        args.n_jobs,
        args.n_estimators,
    )
    canal_model = train_canal_model(
        args.model,
        x_train_all,
        train_all["p_activity"].to_numpy(dtype=float),
        train_all["p_relation"].to_numpy(),
        seed,
        args.n_jobs,
        args.n_estimators,
        args.censor_weight,
        args.censored_anchor_weight,
        args.right_censor_weight,
        args.right_censored_anchor_weight,
    )

    exact_train_pred = exact_model.predict(x_exact)
    fallback_sigma = float(np.std(train_exact["p_activity"].to_numpy(dtype=float) - exact_train_pred))
    fallback_sigma = max(fallback_sigma, 0.05)

    if len(val):
        val_exact_pred = exact_model.predict(x_val_exact)
        val_canal_pred = canal_model.predict(x_val_all)
        base_mae = exact_mae(val, val_exact_pred)
        candidates = []
        for alpha in args.alphas:
            pred = (1.0 - alpha) * val_exact_pred + alpha * val_canal_pred
            sigma = sigma_for(val, pred, fallback_sigma, args.sigma_calibration)
            mixed_nll = mixed_validation_nll(val, pred, sigma)
            mae = exact_mae(val, pred)
            penalty = 0.0
            if np.isfinite(base_mae) and np.isfinite(mae):
                penalty = args.mae_penalty * max(0.0, mae - base_mae - args.mae_tolerance)
            score = mixed_nll + penalty
            candidates.append((score, mixed_nll, mae, sigma, alpha))
        candidates = [c for c in candidates if np.isfinite(c[0])]
        if candidates:
            score, val_mixed_nll, val_mae, sigma, alpha = min(candidates, key=lambda x: x[0])
        else:
            alpha = 1.0
            val_mixed_nll = np.nan
            val_mae = np.nan
            sigma = fallback_sigma
    else:
        alpha = 1.0
        val_mixed_nll = np.nan
        val_mae = np.nan
        sigma = fallback_sigma

    test_exact_pred = exact_model.predict(x_test_exact)
    test_canal_pred = canal_model.predict(x_test_all)
    test_pred = (1.0 - alpha) * test_exact_pred + alpha * test_canal_pred
    row: dict[str, object] = {
        "endpoint": endpoint,
        "target": target,
        "split": split,
        "seed": seed,
        "method": "canal_blend",
        "model": args.model,
        "feature_set": args.feature_set,
        "metadata_mode": args.metadata_mode,
        "sigma_calibration": args.sigma_calibration,
        "status": "ok",
        "train_rows": len(train_all),
        "train_exact_rows": len(train_exact),
        "val_rows": len(val),
        "test_rows": len(test),
        "alpha_canal": alpha,
        "val_mixed_nll": val_mixed_nll,
        "val_exact_mae": val_mae,
        "sigma": sigma,
        "censor_weight": args.censor_weight,
        "censored_anchor_weight": args.censored_anchor_weight,
        "right_censor_weight": args.right_censor_weight if args.right_censor_weight is not None else np.nan,
        "right_censored_anchor_weight": (
            args.right_censored_anchor_weight if args.right_censored_anchor_weight is not None else np.nan
        ),
    }
    row.update(evaluate(test, test_pred, sigma, "test"))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["S1", "S2", "S3", "S4"])
    parser.add_argument("--endpoints", nargs="+", default=["IC50"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2039, 2053])
    parser.add_argument("--model", default="xgb", choices=["xgb", "lgbm"])
    parser.add_argument("--feature-set", default="ecfp_desc", choices=["ecfp", "desc", "ecfp_desc"])
    parser.add_argument(
        "--metadata-mode",
        default="source_light",
        choices=["none", "onehot_assay_doc", "source_light", "source_context", "source_quality"],
    )
    parser.add_argument("--max-targets", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=24)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--censor-weight", type=float, default=0.1)
    parser.add_argument("--censored-anchor-weight", type=float, default=1.0)
    parser.add_argument("--right-censor-weight", type=float, default=None)
    parser.add_argument("--right-censored-anchor-weight", type=float, default=None)
    parser.add_argument("--sigma-calibration", default="mixed_grid", choices=["exact", "mixed_grid"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--mae-penalty", type=float, default=2.0)
    parser.add_argument("--mae-tolerance", type=float, default=0.01)
    parser.add_argument("--tag", default="blend")
    args = parser.parse_args()
    args.n_jobs = min(args.n_jobs, cpu_threads())
    apply_thread_env(args.n_jobs)

    df = pd.read_parquet(args.dataset).reset_index(drop=True)
    selected_targets: dict[str, list[str]] = {}
    keep_mask = pd.Series(False, index=df.index)
    for endpoint in args.endpoints:
        targets = choose_targets(df, endpoint, args.max_targets)
        selected_targets[endpoint] = targets
        keep_mask |= df["endpoint"].eq(endpoint) & df["target_chembl_id"].isin(targets)
    df = df.loc[keep_mask].reset_index(drop=True)
    print(f"[features] building {args.feature_set} once for selected rows={len(df)}", flush=True)
    x_all = featurize(df["canonical_smiles_rdkit"], feature_set=args.feature_set)
    rows = []
    out_dir = ROOT / "results" / "canal_blend"
    out_dir.mkdir(parents=True, exist_ok=True)
    for endpoint in args.endpoints:
        for target in selected_targets[endpoint]:
            for split in args.splits:
                for seed in args.seeds:
                    print(f"[run] endpoint={endpoint} target={target} split={split} seed={seed}", flush=True)
                    try:
                        row = run_one(df, x_all, endpoint, target, split, seed, args)
                    except Exception as exc:
                        row = {
                            "endpoint": endpoint,
                            "target": target,
                            "split": split,
                            "seed": seed,
                            "method": "canal_blend",
                            "model": args.model,
                            "feature_set": args.feature_set,
                            "metadata_mode": args.metadata_mode,
                            "status": "error",
                            "error": repr(exc),
                        }
                    if row:
                        row.setdefault("dataset", args.dataset.stem)
                        rows.append(row)
                        pd.DataFrame(rows).to_csv(out_dir / f"canal_blend_{args.tag}_partial.csv", index=False)
    out_csv = out_dir / f"canal_blend_{args.tag}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    (out_dir / f"canal_blend_{args.tag}.json").write_text(
        json.dumps({"rows": len(rows), "tag": args.tag}, indent=2), encoding="utf-8"
    )
    print(f"[done] {out_csv}", flush=True)


if __name__ == "__main__":
    main()
