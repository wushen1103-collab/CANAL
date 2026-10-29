#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

try:
    from catboost import CatBoostRegressor
except ImportError:  # Optional strong baseline.
    CatBoostRegressor = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from canal.chem import ecfp4
from canal.censored_lgbm import train_censored_lgbm
from canal.censored_xgb import train_censored_xgb
from canal.metrics import (
    censored_direction_accuracy,
    censored_gaussian_nll,
    gaussian_nll,
    interval_coverage,
    regression_metrics,
)
from canal.resources import apply_thread_env, cpu_threads


def rdkit_descriptor_vector(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    values = np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.HeavyAtomMolWt(mol),
            Descriptors.ExactMolWt(mol),
            Descriptors.NumValenceElectrons(mol),
            Descriptors.NumRadicalElectrons(mol),
            Descriptors.BalabanJ(mol),
            Descriptors.BertzCT(mol),
            Descriptors.HallKierAlpha(mol),
            Descriptors.Ipc(mol),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol),
            Descriptors.TPSA(mol),
            Crippen.MolLogP(mol),
            Crippen.MolMR(mol),
            Lipinski.HeavyAtomCount(mol),
            Lipinski.NHOHCount(mol),
            Lipinski.NOCount(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHeteroatoms(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.RingCount(mol),
            Lipinski.NumAromaticRings(mol),
            Lipinski.NumAliphaticRings(mol),
            Lipinski.NumSaturatedRings(mol),
            Lipinski.FractionCSP3(mol),
            rdMolDescriptors.CalcNumAmideBonds(mol),
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            rdMolDescriptors.CalcNumSpiroAtoms(mol),
        ],
        dtype=np.float64,
    )
    values[~np.isfinite(values)] = 0.0
    values = np.clip(values, -1e6, 1e6)
    return values.astype(np.float32)


def featurize(smiles: pd.Series, n_bits: int = 2048, feature_set: str = "ecfp") -> sparse.csr_matrix:
    rows = []
    keep = []
    for i, smi in enumerate(smiles):
        smi = str(smi)
        fp = ecfp4(smi, n_bits=n_bits) if feature_set in {"ecfp", "ecfp_desc"} else None
        desc = rdkit_descriptor_vector(smi) if feature_set in {"desc", "ecfp_desc"} else None
        if feature_set == "ecfp":
            feat = fp
        elif feature_set == "desc":
            feat = desc
        elif feature_set == "ecfp_desc":
            feat = np.concatenate([fp, desc]).astype(np.float32) if fp is not None and desc is not None else None
        else:
            raise ValueError(feature_set)
        if feat is None:
            continue
        rows.append(feat)
        keep.append(i)
    if len(rows) != len(smiles):
        raise ValueError(f"Featurization failed for {len(smiles) - len(rows)} molecules.")
    return sparse.csr_matrix(np.vstack(rows).astype(np.float32))


def choose_targets(df: pd.DataFrame, endpoint: str, max_targets: int | None) -> list[str]:
    sub = df[df["endpoint"] == endpoint]
    counts = sub.groupby("target_chembl_id").size().sort_values(ascending=False)
    if max_targets:
        counts = counts.head(max_targets)
    return counts.index.tolist()


def filter_requested_panel(
    df: pd.DataFrame, endpoints: list[str], max_targets: int | None
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    targets_by_endpoint = {endpoint: choose_targets(df, endpoint, max_targets) for endpoint in endpoints}
    masks = []
    for endpoint, targets in targets_by_endpoint.items():
        if targets:
            masks.append(df["endpoint"].eq(endpoint) & df["target_chembl_id"].isin(targets))
    if not masks:
        return df.iloc[0:0].copy(), targets_by_endpoint
    mask = masks[0]
    for next_mask in masks[1:]:
        mask = mask | next_mask
    return df[mask].reset_index(drop=True), targets_by_endpoint


def train_model(model_name: str, x_train, y_train, seed: int, n_jobs: int, n_estimators: int):
    if model_name == "xgb":
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
            reg_lambda=1.0,
        )
    elif model_name == "rf":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=n_jobs,
        )
    elif model_name == "etr":
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=n_jobs,
        )
    elif model_name == "ridge":
        model = Ridge(alpha=1.0, solver="lsqr", tol=1e-4)
    elif model_name == "lgbm":
        model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.04,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="regression",
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
    elif model_name == "catboost":
        if CatBoostRegressor is None:
            raise ImportError("catboost is not installed; install it to run --model catboost")
        model = CatBoostRegressor(
            iterations=n_estimators,
            depth=8,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=seed,
            thread_count=n_jobs,
            allow_writing_files=False,
            verbose=False,
        )
    else:
        raise ValueError(model_name)
    model.fit(x_train, y_train)
    return model


def present_flag(frame: pd.DataFrame, col: str) -> np.ndarray:
    if col not in frame:
        return np.zeros((len(frame), 1), dtype=np.float32)
    text = frame[col].fillna("").astype(str).str.strip().str.lower()
    flag = text.ne("") & text.ne("none") & text.ne("nan")
    return flag.to_numpy(dtype=np.float32).reshape(-1, 1)


def source_numeric_features(frame: pd.DataFrame, year_center: float) -> np.ndarray:
    confidence = pd.to_numeric(frame.get("confidence_score", pd.Series(index=frame.index)), errors="coerce")
    duplicate = pd.to_numeric(frame.get("potential_duplicate", pd.Series(index=frame.index)), errors="coerce")
    year = pd.to_numeric(frame.get("document_year", pd.Series(index=frame.index)), errors="coerce")
    year_missing = year.isna().to_numpy(dtype=np.float32).reshape(-1, 1)
    year_scaled = ((year.fillna(year_center) - year_center) / 10.0).to_numpy(dtype=np.float32).reshape(-1, 1)
    return np.hstack(
        [
            (confidence.fillna(-1.0) / 10.0).to_numpy(dtype=np.float32).reshape(-1, 1),
            duplicate.fillna(0.0).to_numpy(dtype=np.float32).reshape(-1, 1),
            year_scaled,
            year_missing,
            present_flag(frame, "data_validity_comment"),
            present_flag(frame, "activity_comment"),
        ]
    )


def with_metadata_features(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, x_train, x_val, x_test, mode: str):
    if mode == "none":
        return x_train, x_val, x_test
    if mode == "onehot_assay_doc":
        cols = ["assay_chembl_id", "document_chembl_id"]
    elif mode in {"source_light", "source_context"}:
        cols = ["assay_type", "assay_organism"]
    elif mode == "source_quality":
        cols = []
    else:
        raise ValueError(mode)
    if cols:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
        train_meta = train[cols].fillna("NA").astype(str)
        enc_train = enc.fit_transform(train_meta)
        enc_val = enc.transform(val[cols].fillna("NA").astype(str)) if len(val) else None
        enc_test = enc.transform(test[cols].fillna("NA").astype(str))
    else:
        enc_train = sparse.csr_matrix((len(train), 0), dtype=np.float32)
        enc_val = sparse.csr_matrix((len(val), 0), dtype=np.float32) if len(val) else None
        enc_test = sparse.csr_matrix((len(test), 0), dtype=np.float32)
    if mode in {"source_light", "source_quality"}:
        year = pd.to_numeric(train.get("document_year", pd.Series(index=train.index)), errors="coerce")
        year_center = float(year.median()) if year.notna().any() else 2010.0
        enc_train = sparse.hstack([enc_train, sparse.csr_matrix(source_numeric_features(train, year_center))], format="csr")
        if len(val):
            enc_val = sparse.hstack([enc_val, sparse.csr_matrix(source_numeric_features(val, year_center))], format="csr")
        enc_test = sparse.hstack([enc_test, sparse.csr_matrix(source_numeric_features(test, year_center))], format="csr")
    x_train = sparse.hstack([x_train, enc_train], format="csr")
    if x_val is not None:
        x_val = sparse.hstack([x_val, enc_val], format="csr")
    x_test = sparse.hstack([x_test, enc_test], format="csr")
    return x_train, x_val, x_test


def fit_assay_shrink_bias(train: pd.DataFrame, train_pred: np.ndarray, shrink_lambda: float) -> dict[str, float]:
    exact = train["p_relation"].eq("=")
    if exact.sum() == 0:
        return {}
    residual = train.loc[exact, "p_activity"].to_numpy(dtype=float) - train_pred[exact.to_numpy()]
    tmp = pd.DataFrame(
        {
            "assay": train.loc[exact, "assay_chembl_id"].fillna("NA").astype(str).to_numpy(),
            "residual": residual,
        }
    )
    grouped = tmp.groupby("assay")["residual"].agg(["sum", "count"])
    bias = grouped["sum"] / (grouped["count"] + float(shrink_lambda))
    return bias.to_dict()


def apply_assay_bias(df_split: pd.DataFrame, pred: np.ndarray, bias: dict[str, float]) -> np.ndarray:
    if not bias:
        return pred
    offsets = df_split["assay_chembl_id"].fillna("NA").astype(str).map(bias).fillna(0.0).to_numpy(dtype=float)
    return pred + offsets


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


def mixed_validation_nll(df_split: pd.DataFrame, pred: np.ndarray, sigma: float) -> float:
    exact = df_split["p_relation"].eq("=")
    parts = []
    weights = []
    if exact.any():
        nll = gaussian_nll(df_split.loc[exact, "p_activity"].to_numpy(), pred[exact.to_numpy()], sigma)
        if np.isfinite(nll):
            parts.append(nll)
            weights.append(int(exact.sum()))
    cens = ~exact
    if cens.any():
        nll = censored_gaussian_nll(
            df_split.loc[cens, "p_activity"].to_numpy(),
            df_split.loc[cens, "p_relation"].to_numpy(),
            pred[cens.to_numpy()],
            sigma,
        )
        if np.isfinite(nll):
            parts.append(nll)
            weights.append(int(cens.sum()))
    if not parts:
        return np.inf
    return float(np.average(parts, weights=weights))


def calibrate_sigma(df_val: pd.DataFrame, val_pred: np.ndarray, fallback_sigma: float, mode: str) -> float:
    fallback_sigma = max(float(fallback_sigma), 0.05)
    if mode == "exact":
        return fallback_sigma
    if mode != "mixed_grid" or len(df_val) == 0:
        return fallback_sigma
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.1, 1.5, 29),
                np.linspace(1.6, 3.0, 8),
                np.array([fallback_sigma]),
            ]
        )
    )
    scores = [(mixed_validation_nll(df_val, val_pred, sigma), sigma) for sigma in grid]
    scores = [(score, sigma) for score, sigma in scores if np.isfinite(score)]
    if not scores:
        return fallback_sigma
    return float(min(scores, key=lambda x: x[0])[1])


def run_one(
    df: pd.DataFrame,
    x_all: sparse.csr_matrix,
    endpoint: str,
    target: str,
    split: str,
    seed: int,
    treatment: str,
    model_name: str,
    n_jobs: int,
    n_estimators: int,
    feature_set: str,
    censor_weight: float,
    censored_anchor_weight: float,
    right_censor_weight: float | None,
    right_censored_anchor_weight: float | None,
    metadata_mode: str,
    sigma_calibration: str,
    posthoc_bias: str,
    posthoc_lambda: float,
) -> dict[str, object] | None:
    split_col = f"split_{split.lower()}"
    sub = df[(df["endpoint"] == endpoint) & (df["target_chembl_id"] == target)].copy()
    if sub.empty or split_col not in sub:
        return None

    train = sub[sub[split_col] == "train"].copy()
    val = sub[sub[split_col] == "val"].copy()
    test = sub[sub[split_col] == "test"].copy()

    if treatment in {"exact_only", "discard_censored"}:
        train = train[train["p_relation"] == "="]
    elif treatment in {"censored_as_exact", "censored_likelihood"}:
        pass
    else:
        raise ValueError(treatment)

    if treatment == "discard_censored":
        val = val[val["p_relation"] == "="]
        test = test[test["p_relation"] == "="]

    if len(train) < 100 or len(test) < 30:
        return {
            "endpoint": endpoint,
            "target": target,
            "split": split,
            "seed": seed,
            "treatment": treatment,
            "model": model_name,
            "feature_set": feature_set,
            "status": "too_few_rows",
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
        }

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
            raise ValueError("censored_likelihood is currently implemented only for model=xgb or model=lgbm")
    else:
        model = train_model(model_name, x_train, y_train, seed, n_jobs, n_estimators)

    if posthoc_bias == "assay_shrink":
        train_pred_for_bias = model.predict(x_train)
        assay_bias = fit_assay_shrink_bias(train, train_pred_for_bias, posthoc_lambda)
    elif posthoc_bias == "none":
        assay_bias = {}
    else:
        raise ValueError(posthoc_bias)

    if x_val is not None and len(val):
        val_pred = apply_assay_bias(val, model.predict(x_val), assay_bias)
        exact_val = val["p_relation"].eq("=")
        if exact_val.sum() >= 20:
            sigma = float(np.std(val.loc[exact_val, "p_activity"].to_numpy() - val_pred[exact_val.to_numpy()]))
        else:
            sigma = float(np.std(y_train - model.predict(x_train)))
        sigma = calibrate_sigma(val, val_pred, sigma, sigma_calibration)
    else:
        sigma = float(np.std(y_train - model.predict(x_train)))
    sigma = max(sigma, 0.05)

    test_pred = apply_assay_bias(test, model.predict(x_test), assay_bias)

    row: dict[str, object] = {
        "endpoint": endpoint,
        "target": target,
        "split": split,
        "seed": seed,
        "treatment": treatment,
        "model": model_name,
        "feature_set": feature_set,
        "metadata_mode": metadata_mode,
        "sigma_calibration": sigma_calibration,
        "posthoc_bias": posthoc_bias,
        "posthoc_lambda": posthoc_lambda if posthoc_bias != "none" else np.nan,
        "status": "ok",
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "sigma": sigma,
        "censor_weight": censor_weight if treatment == "censored_likelihood" else np.nan,
        "censored_anchor_weight": censored_anchor_weight if treatment == "censored_likelihood" else np.nan,
        "right_censor_weight": (
            right_censor_weight if treatment == "censored_likelihood" and right_censor_weight is not None else np.nan
        ),
        "right_censored_anchor_weight": (
            right_censored_anchor_weight
            if treatment == "censored_likelihood" and right_censored_anchor_weight is not None
            else np.nan
        ),
        "n_assays_train": int(train["assay_chembl_id"].nunique()),
        "n_assays_test": int(test["assay_chembl_id"].nunique()),
    }
    row.update(evaluate(test, test_pred, sigma, "test"))

    worst_source = (
        pd.DataFrame({"assay": test["assay_chembl_id"].to_numpy(), "exact": test["p_relation"].eq("=").to_numpy(), "y": test["p_activity"].to_numpy(), "pred": test_pred})
        .query("exact")
        .groupby("assay")
        .apply(lambda g: np.mean(np.abs(g["pred"] - g["y"])) if len(g) >= 10 else np.nan, include_groups=False)
        .dropna()
    )
    row["worst_assay_mae"] = float(worst_source.max()) if len(worst_source) else np.nan
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["S1", "S2", "S3"])
    parser.add_argument("--endpoints", nargs="+", default=["IC50", "Ki"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2039, 2053])
    parser.add_argument(
        "--treatments",
        nargs="+",
        default=["exact_only", "discard_censored", "censored_as_exact", "censored_likelihood"],
    )
    parser.add_argument("--model", default="xgb", choices=["xgb", "rf", "etr", "ridge", "lgbm", "catboost"])
    parser.add_argument("--feature-set", default="ecfp", choices=["ecfp", "desc", "ecfp_desc"])
    parser.add_argument("--max-targets", type=int, default=3)
    parser.add_argument(
        "--target-ids",
        nargs="*",
        default=None,
        help="Optional explicit target IDs; when supplied, --max-targets is ignored.",
    )
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--censor-weight", type=float, default=1.0)
    parser.add_argument("--censored-anchor-weight", type=float, default=0.0)
    parser.add_argument("--right-censor-weight", type=float, default=None)
    parser.add_argument("--right-censored-anchor-weight", type=float, default=None)
    parser.add_argument(
        "--metadata-mode",
        default="none",
        choices=["none", "onehot_assay_doc", "source_light", "source_context", "source_quality"],
    )
    parser.add_argument("--sigma-calibration", default="exact", choices=["exact", "mixed_grid"])
    parser.add_argument("--posthoc-bias", default="none", choices=["none", "assay_shrink"])
    parser.add_argument("--posthoc-lambda", type=float, default=20.0)
    args = parser.parse_args()
    args.n_jobs = min(args.n_jobs, cpu_threads())
    apply_thread_env(args.n_jobs)
    print(f"[resource] n_jobs={args.n_jobs} cpu_cap=192")

    raw_df = pd.read_parquet(args.dataset).reset_index(drop=True)
    if args.target_ids:
        raw_df = raw_df[raw_df["target_chembl_id"].isin(args.target_ids)].reset_index(drop=True)
    df, targets_by_endpoint = filter_requested_panel(
        raw_df, args.endpoints, None if args.target_ids else args.max_targets
    )
    print(
        "[dataset] selected "
        f"{len(df)} / {len(raw_df)} rows for endpoints={args.endpoints} max_targets={args.max_targets}"
    )
    print(f"[features] building {args.feature_set} once for {len(df)} rows")
    x_all = featurize(df["canonical_smiles_rdkit"], feature_set=args.feature_set)
    out_dir = ROOT / "results" / "problem_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for endpoint in args.endpoints:
        targets = targets_by_endpoint.get(endpoint, [])
        for target in targets:
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
                                args.model,
                                args.n_jobs,
                                args.n_estimators,
                                args.feature_set,
                                args.censor_weight,
                                args.censored_anchor_weight,
                                args.right_censor_weight,
                                args.right_censored_anchor_weight,
                                args.metadata_mode,
                                args.sigma_calibration,
                                args.posthoc_bias,
                                args.posthoc_lambda,
                            )
                        except Exception as exc:
                            row = {
                                "endpoint": endpoint,
                                "target": target,
                                "split": split,
                                "seed": seed,
                                "treatment": treatment,
                                "model": args.model,
                                "status": "error",
                                "error": repr(exc),
                            }
                        if row:
                            row.setdefault("dataset", args.dataset.stem)
                            rows.append(row)
                            partial_tag = args.tag or f"{args.dataset.stem}_{args.model}"
                            pd.DataFrame(rows).to_csv(
                                out_dir / f"problem_check_partial_{partial_tag}.csv", index=False
                            )

    result = pd.DataFrame(rows)
    tag = f"_{args.tag}" if args.tag else ""
    out_csv = out_dir / f"problem_check_{args.dataset.stem}_{args.model}{tag}.csv"
    result.to_csv(out_csv, index=False)
    summary = {
        "dataset": str(args.dataset),
        "rows": len(result),
        "ok_rows": int((result.get("status") == "ok").sum()) if len(result) else 0,
        "splits": args.splits,
        "endpoints": args.endpoints,
        "seeds": args.seeds,
        "treatments": args.treatments,
    }
    (out_dir / f"problem_check_{args.dataset.stem}_{args.model}{tag}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[done] {out_csv}")


if __name__ == "__main__":
    main()
