#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from canal.metrics import censored_direction_accuracy, censored_gaussian_nll, gaussian_nll, interval_coverage, regression_metrics
from canal.resources import apply_thread_env, cpu_threads
from canal.tobit import standardize_train_val_test, train_linear_tobit
from run_problem_check import (
    calibrate_sigma,
    featurize,
    filter_requested_panel,
    with_metadata_features,
)


def evaluate(df_split: pd.DataFrame, pred: np.ndarray, sigma: float, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    exact = df_split["p_relation"].eq("=")
    if exact.any():
        reg = regression_metrics(df_split.loc[exact, "p_activity"].to_numpy(), pred[exact.to_numpy()])
        out.update({f"{prefix}_{k}": v for k, v in reg.items()})
        out[f"{prefix}_nll"] = gaussian_nll(df_split.loc[exact, "p_activity"].to_numpy(), pred[exact.to_numpy()], sigma)
        out[f"{prefix}_coverage90"] = interval_coverage(
            df_split.loc[exact, "p_activity"].to_numpy(), pred[exact.to_numpy()], sigma
        )
    else:
        for key in ["n", "mae", "rmse", "spearman", "nll", "coverage90"]:
            out[f"{prefix}_{key}"] = np.nan
    cens = ~exact
    if cens.any():
        out[f"{prefix}_censored_n"] = int(cens.sum())
        out[f"{prefix}_censored_nll"] = censored_gaussian_nll(
            df_split.loc[cens, "p_activity"].to_numpy(),
            df_split.loc[cens, "p_relation"].to_numpy(),
            pred[cens.to_numpy()],
            sigma,
        )
        out[f"{prefix}_censored_dir_acc"] = censored_direction_accuracy(
            df_split.loc[cens, "p_activity"].to_numpy(),
            df_split.loc[cens, "p_relation"].to_numpy(),
            pred[cens.to_numpy()],
        )
    else:
        out[f"{prefix}_censored_n"] = 0
        out[f"{prefix}_censored_nll"] = np.nan
        out[f"{prefix}_censored_dir_acc"] = np.nan
    return out


def worst_assay_mae(test: pd.DataFrame, pred: np.ndarray) -> float:
    exact = test["p_relation"].eq("=").to_numpy()
    if not exact.any():
        return np.nan
    source = pd.DataFrame(
        {
            "assay": test["assay_chembl_id"].to_numpy(),
            "exact": exact,
            "y": test["p_activity"].to_numpy(dtype=float),
            "pred": pred,
        }
    )
    grouped = (
        source.query("exact")
        .groupby("assay")
        .apply(lambda g: np.mean(np.abs(g["pred"] - g["y"])) if len(g) >= 10 else np.nan, include_groups=False)
        .dropna()
    )
    return float(grouped.max()) if len(grouped) else np.nan


def fit_exact_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float):
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-4)
    model.fit(x_train, y_train)
    return model


def run_one(
    df: pd.DataFrame,
    x_all,
    endpoint: str,
    target: str,
    split: str,
    seed: int,
    treatment: str,
    feature_set: str,
    metadata_mode: str,
    sigma_calibration: str,
    ridge_alpha: float,
    max_iter: int,
) -> dict[str, object] | None:
    split_col = f"split_{split.lower()}"
    sub = df[(df["endpoint"] == endpoint) & (df["target_chembl_id"] == target)].copy()
    if sub.empty or split_col not in sub:
        return None

    train = sub[sub[split_col] == "train"].copy()
    val = sub[sub[split_col] == "val"].copy()
    test = sub[sub[split_col] == "test"].copy()
    if treatment in {"tobit_exact_only", "tobit_standard"}:
        pass
    else:
        raise ValueError(treatment)
    train_fit = train[train["p_relation"].eq("=")].copy() if treatment == "tobit_exact_only" else train

    if len(train_fit) < 100 or len(test) < 30:
        return {
            "endpoint": endpoint,
            "target": target,
            "split": split,
            "seed": seed,
            "treatment": treatment,
            "model": "tobit",
            "feature_set": feature_set,
            "metadata_mode": metadata_mode,
            "status": "too_few_rows",
            "train_rows": len(train_fit),
            "val_rows": len(val),
            "test_rows": len(test),
        }

    x_train = x_all[train_fit.index.to_numpy()]
    y_train = train_fit["p_activity"].to_numpy(dtype=float)
    x_val = x_all[val.index.to_numpy()] if len(val) else None
    x_test = x_all[test.index.to_numpy()]
    x_train, x_val, x_test = with_metadata_features(train_fit, val, test, x_train, x_val, x_test, metadata_mode)
    x_train, x_val, x_test = standardize_train_val_test(x_train, x_val, x_test)

    if treatment == "tobit_standard":
        model = train_linear_tobit(
            x_train,
            y_train,
            train_fit["p_relation"].to_numpy(),
            ridge_alpha=ridge_alpha,
            max_iter=max_iter,
        )
    else:
        model = fit_exact_ridge(x_train, y_train, alpha=ridge_alpha)

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

    test_pred = model.predict(x_test)
    row: dict[str, object] = {
        "endpoint": endpoint,
        "target": target,
        "split": split,
        "seed": seed,
        "treatment": treatment,
        "model": "tobit",
        "feature_set": feature_set,
        "metadata_mode": metadata_mode,
        "sigma_calibration": sigma_calibration,
        "posthoc_bias": "none",
        "posthoc_lambda": np.nan,
        "status": "ok",
        "train_rows": len(train_fit),
        "val_rows": len(val),
        "test_rows": len(test),
        "sigma": sigma,
        "censor_weight": 1.0 if treatment == "tobit_standard" else np.nan,
        "censored_anchor_weight": 0.0 if treatment == "tobit_standard" else np.nan,
        "right_censor_weight": np.nan,
        "right_censored_anchor_weight": np.nan,
        "tobit_ridge_alpha": ridge_alpha,
        "tobit_max_iter": max_iter if treatment == "tobit_standard" else np.nan,
        "n_assays_train": int(train_fit["assay_chembl_id"].nunique()),
        "n_assays_test": int(test["assay_chembl_id"].nunique()),
    }
    row.update(evaluate(test, test_pred, sigma, "test"))
    row["worst_assay_mae"] = worst_assay_mae(test, test_pred)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["S1", "S2", "S3", "S4"])
    parser.add_argument("--endpoints", nargs="+", default=["IC50"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2039, 2053, 2069, 2081])
    parser.add_argument("--treatments", nargs="+", default=["tobit_exact_only", "tobit_standard"])
    parser.add_argument("--feature-set", default="desc", choices=["desc", "ecfp_desc"])
    parser.add_argument("--metadata-mode", default="source_light", choices=["none", "source_light", "source_quality", "source_context"])
    parser.add_argument("--max-targets", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--sigma-calibration", default="mixed_grid", choices=["exact", "mixed_grid"])
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tag", default="standard_tobit_5seed")
    args = parser.parse_args()

    args.n_jobs = min(args.n_jobs, cpu_threads())
    apply_thread_env(args.n_jobs)
    print(f"[resource] n_jobs={args.n_jobs} cpu_cap=192")
    raw_df = pd.read_parquet(args.dataset).reset_index(drop=True)
    df, targets_by_endpoint = filter_requested_panel(raw_df, args.endpoints, args.max_targets)
    print(
        "[dataset] selected "
        f"{len(df)} / {len(raw_df)} rows for endpoints={args.endpoints} max_targets={args.max_targets}"
    )
    print(f"[features] building {args.feature_set} once for {len(df)} rows")
    x_all = featurize(df["canonical_smiles_rdkit"], feature_set=args.feature_set)

    rows = []
    out_dir = ROOT / "results" / "problem_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    for endpoint in args.endpoints:
        for target in targets_by_endpoint.get(endpoint, []):
            for split in args.splits:
                for seed in args.seeds:
                    for treatment in args.treatments:
                        print(f"[run] endpoint={endpoint} target={target} split={split} seed={seed} treatment={treatment}")
                        try:
                            row = run_one(
                                df,
                                x_all,
                                endpoint,
                                target,
                                split,
                                seed,
                                treatment,
                                args.feature_set,
                                args.metadata_mode,
                                args.sigma_calibration,
                                args.ridge_alpha,
                                args.max_iter,
                            )
                        except Exception as exc:
                            row = {
                                "endpoint": endpoint,
                                "target": target,
                                "split": split,
                                "seed": seed,
                                "treatment": treatment,
                                "model": "tobit",
                                "feature_set": args.feature_set,
                                "metadata_mode": args.metadata_mode,
                                "status": "error",
                                "error": repr(exc),
                            }
                        if row:
                            row.setdefault("dataset", args.dataset.stem)
                            rows.append(row)
                            pd.DataFrame(rows).to_csv(out_dir / "problem_check_tobit_partial.csv", index=False)

    result = pd.DataFrame(rows)
    out_csv = out_dir / f"problem_check_{args.dataset.stem}_tobit_{args.tag}.csv"
    result.to_csv(out_csv, index=False)
    summary = {
        "dataset": str(args.dataset),
        "rows": len(result),
        "ok_rows": int((result.get("status") == "ok").sum()) if len(result) else 0,
        "splits": args.splits,
        "endpoints": args.endpoints,
        "seeds": args.seeds,
        "treatments": args.treatments,
        "feature_set": args.feature_set,
        "metadata_mode": args.metadata_mode,
        "ridge_alpha": args.ridge_alpha,
    }
    out_csv.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] {out_csv}")


if __name__ == "__main__":
    main()
