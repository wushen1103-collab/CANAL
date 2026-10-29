#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from canal.censored_lgbm import train_censored_lgbm
from canal.metrics import censored_direction_accuracy, censored_gaussian_nll, gaussian_nll, regression_metrics
from canal.resources import apply_thread_env, cpu_threads
from run_problem_check import featurize, filter_requested_panel, train_model, with_metadata_features


def evaluate_block(df_split: pd.DataFrame, pred: np.ndarray, sigma: float, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    exact = df_split["p_relation"].eq("=")
    if exact.any():
        reg = regression_metrics(df_split.loc[exact, "p_activity"].to_numpy(), pred[exact.to_numpy()])
        out.update({f"{prefix}_{k}": v for k, v in reg.items()})
        out[f"{prefix}_nll"] = gaussian_nll(df_split.loc[exact, "p_activity"].to_numpy(), pred[exact.to_numpy()], sigma)
    else:
        for key in ["n", "mae", "rmse", "spearman", "nll"]:
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


def candidate_grid(names: set[str] | None = None) -> list[dict[str, object]]:
    grid = [
        {"candidate": "exact_only", "treatment": "exact_only"},
        {"candidate": "censored_as_exact", "treatment": "censored_as_exact"},
        {"candidate": "pure_censored_cw1_aw0", "treatment": "censored_likelihood", "cw": 1.0, "aw": 0.0},
        {"candidate": "canal_cw0p1_aw1", "treatment": "censored_likelihood", "cw": 0.1, "aw": 1.0},
        {"candidate": "canal_cw0p25_aw0p5", "treatment": "censored_likelihood", "cw": 0.25, "aw": 0.5},
        {"candidate": "canal_cw0p25_aw1", "treatment": "censored_likelihood", "cw": 0.25, "aw": 1.0},
        {"candidate": "canal_cw0p5_aw0p5", "treatment": "censored_likelihood", "cw": 0.5, "aw": 0.5},
        {"candidate": "canal_cw0p5_aw1", "treatment": "censored_likelihood", "cw": 0.5, "aw": 1.0},
        {
            "candidate": "relation_cw0p25_aw1_right0p1_aw0",
            "treatment": "censored_likelihood",
            "cw": 0.25,
            "aw": 1.0,
            "right_cw": 0.1,
            "right_aw": 0.0,
        },
    ]
    if names:
        grid = [row for row in grid if str(row["candidate"]) in names]
    return grid


def train_predict_candidate(
    x_train,
    y_train: np.ndarray,
    relation_train: np.ndarray,
    x_val,
    x_test,
    candidate: dict[str, object],
    seed: int,
    n_jobs: int,
    n_estimators: int,
):
    treatment = str(candidate["treatment"])
    if treatment == "censored_likelihood":
        model = train_censored_lgbm(
            x_train,
            y_train,
            relation_train,
            seed=seed,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            censor_weight=float(candidate.get("cw", 1.0)),
            censored_anchor_weight=float(candidate.get("aw", 0.0)),
            right_censor_weight=candidate.get("right_cw"),
            right_censored_anchor_weight=candidate.get("right_aw"),
        )
    else:
        model = train_model("lgbm", x_train, y_train, seed, n_jobs, n_estimators)
    return model.predict(x_val), model.predict(x_test), model.predict(x_train)


def run_one(
    df: pd.DataFrame,
    x_all: sparse.csr_matrix,
    endpoint: str,
    target: str,
    split: str,
    seed: int,
    candidate: dict[str, object],
    n_jobs: int,
    n_estimators: int,
    metadata_mode: str,
) -> dict[str, object] | None:
    split_col = f"split_{split.lower()}"
    sub = df[(df["endpoint"].eq(endpoint)) & (df["target_chembl_id"].eq(target))].copy()
    if sub.empty or split_col not in sub:
        return None
    train = sub[sub[split_col].eq("train")].copy()
    val = sub[sub[split_col].eq("val")].copy()
    test = sub[sub[split_col].eq("test")].copy()
    treatment = str(candidate["treatment"])
    if treatment == "exact_only":
        train = train[train["p_relation"].eq("=")].copy()
    if len(train) < 100 or len(val) < 30 or len(test) < 30:
        return {
            "endpoint": endpoint,
            "target": target,
            "split": split,
            "seed": seed,
            "candidate": candidate["candidate"],
            "status": "too_few_rows",
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
        }
    x_train = x_all[train.index.to_numpy()]
    y_train = train["p_activity"].to_numpy(dtype=float)
    x_val = x_all[val.index.to_numpy()]
    x_test = x_all[test.index.to_numpy()]
    x_train, x_val, x_test = with_metadata_features(train, val, test, x_train, x_val, x_test, metadata_mode)
    val_pred, test_pred, train_pred = train_predict_candidate(
        x_train,
        y_train,
        train["p_relation"].to_numpy(),
        x_val,
        x_test,
        candidate,
        seed,
        n_jobs,
        n_estimators,
    )
    exact_val = val["p_relation"].eq("=")
    if exact_val.sum() >= 20:
        sigma = float(np.std(val.loc[exact_val, "p_activity"].to_numpy() - val_pred[exact_val.to_numpy()]))
    else:
        sigma = float(np.std(y_train - train_pred))
    sigma = max(sigma, 0.05)

    row: dict[str, object] = {
        "endpoint": endpoint,
        "target": target,
        "split": split,
        "seed": seed,
        "candidate": candidate["candidate"],
        "status": "ok",
        "model": "lgbm",
        "feature_set": "ecfp_desc",
        "metadata_mode": metadata_mode,
        "treatment": treatment,
        "censor_weight": candidate.get("cw", np.nan),
        "censored_anchor_weight": candidate.get("aw", np.nan),
        "right_censor_weight": candidate.get("right_cw", np.nan),
        "right_censored_anchor_weight": candidate.get("right_aw", np.nan),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "sigma": sigma,
    }
    row.update(evaluate_block(val, val_pred, sigma, "val"))
    row.update(evaluate_block(test, test_pred, sigma, "test"))
    return row


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    ok = rows[rows["status"].eq("ok")].copy()
    group_cols = [
        "candidate",
        "model",
        "feature_set",
        "metadata_mode",
        "treatment",
        "censor_weight",
        "censored_anchor_weight",
        "right_censor_weight",
        "right_censored_anchor_weight",
    ]
    out = []
    for keys, group in ok.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["runs"] = len(group)
        row["targets"] = int(group["target"].nunique())
        row["splits"] = int(group["split"].nunique())
        row["seeds"] = int(group["seed"].nunique())
        for metric in [
            "val_mae",
            "val_censored_nll",
            "val_censored_dir_acc",
            "test_mae",
            "test_censored_nll",
            "test_censored_dir_acc",
        ]:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
        out.append(row)
    result = pd.DataFrame(out)
    if result.empty:
        return result
    exact_val_mae = result.loc[result["candidate"].eq("exact_only"), "val_mae_mean"]
    baseline_mae = float(exact_val_mae.iloc[0]) if len(exact_val_mae) else float(result["val_mae_mean"].min())
    result["val_mae_delta_vs_exact"] = result["val_mae_mean"] - baseline_mae
    result["val_boundary_score"] = result["val_censored_nll_mean"] - result["val_censored_dir_acc_mean"]
    result["val_mae_constrained"] = result["val_mae_delta_vs_exact"].le(0.04)
    result["selection_rank"] = (
        result.assign(_penalty=np.where(result["val_mae_constrained"], 0.0, 100.0))
        .sort_values(["_penalty", "val_boundary_score", "val_mae_mean"])
        .reset_index()
        .reset_index()
        .set_index("index")["level_0"]
        + 1
    )
    return result.sort_values(["selection_rank", "val_boundary_score"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/chembl35_ic50_top20_v2.parquet"))
    parser.add_argument("--splits", nargs="+", default=["S1", "S2", "S3", "S4"])
    parser.add_argument("--endpoints", nargs="+", default=["IC50"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027])
    parser.add_argument("--max-targets", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--metadata-mode", default="source_light")
    parser.add_argument("--candidates", nargs="*", default=None)
    parser.add_argument("--out-prefix", type=Path, default=Path("results/validation_sensitivity/ic50_lgbm_validation_sensitivity_s2027"))
    args = parser.parse_args()

    args.n_jobs = min(args.n_jobs, cpu_threads())
    apply_thread_env(args.n_jobs)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_parquet(args.dataset).reset_index(drop=True)
    df, targets_by_endpoint = filter_requested_panel(raw_df, args.endpoints, args.max_targets)
    print(f"[resource] n_jobs={args.n_jobs}")
    print(f"[dataset] selected {len(df)} / {len(raw_df)} rows")
    x_all = featurize(df["canonical_smiles_rdkit"], feature_set="ecfp_desc")
    rows = []
    grid = candidate_grid(set(args.candidates) if args.candidates else None)
    for endpoint in args.endpoints:
        for target in targets_by_endpoint.get(endpoint, []):
            for split in args.splits:
                for seed in args.seeds:
                    for candidate in grid:
                        print(f"[run] endpoint={endpoint} target={target} split={split} seed={seed} candidate={candidate['candidate']}")
                        try:
                            row = run_one(
                                df,
                                x_all,
                                endpoint,
                                target,
                                split,
                                seed,
                                candidate,
                                args.n_jobs,
                                args.n_estimators,
                                args.metadata_mode,
                            )
                        except Exception as exc:
                            row = {
                                "endpoint": endpoint,
                                "target": target,
                                "split": split,
                                "seed": seed,
                                "candidate": candidate["candidate"],
                                "status": "error",
                                "error": repr(exc),
                            }
                        if row:
                            rows.append(row)
                            pd.DataFrame(rows).to_csv(args.out_prefix.with_suffix(".partial.csv"), index=False)
    result = pd.DataFrame(rows)
    result.to_csv(args.out_prefix.with_suffix(".csv"), index=False)
    summary = summarize(result)
    summary.to_csv(args.out_prefix.with_suffix(".summary.csv"), index=False)
    args.out_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "rows": len(result),
                "ok_rows": int(result["status"].eq("ok").sum()) if len(result) else 0,
                "splits": args.splits,
                "seeds": args.seeds,
                "max_targets": args.max_targets,
                "candidates": [c["candidate"] for c in grid],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] wrote {args.out_prefix.with_suffix('.summary.csv')}")


if __name__ == "__main__":
    main()
