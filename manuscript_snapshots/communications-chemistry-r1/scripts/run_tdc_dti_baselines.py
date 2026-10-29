#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from canal.chem import ecfp4
from canal.metrics import regression_metrics
from canal.resources import apply_thread_env, cpu_threads


def load_tdc(name: str, path: Path) -> pd.DataFrame:
    try:
        from tdc.multi_pred import DTI
    except ImportError as exc:
        raise SystemExit("PyTDC is required. Install PyTDC in the CANAL environment first.") from exc
    return DTI(name=name, path=str(path)).get_data().reset_index(drop=True)


def get_tdc_split(name: str, path: Path, split: str, seed: int, frac: list[float]) -> dict[str, pd.DataFrame]:
    from tdc.multi_pred import DTI

    data = DTI(name=name, path=str(path))
    if split == "random":
        return data.get_split(method="random", seed=seed, frac=frac)
    if split == "cold_drug":
        return data.get_split(method="cold_split", column_name="Drug", seed=seed, frac=frac)
    if split == "cold_target":
        return data.get_split(method="cold_split", column_name="Target", seed=seed, frac=frac)
    if split == "cold_both":
        return data.get_split(method="cold_split", column_name=["Drug", "Target"], seed=seed, frac=frac)
    raise ValueError(split)


def standardize_label(y: pd.Series, dataset: str, label_space: str) -> tuple[np.ndarray, str]:
    values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    if label_space == "auto":
        label_space = "pvalue" if dataset.lower() in {"davis", "bindingdb_kd"} else "raw"
    if label_space == "pvalue":
        values = np.where(values > 0, 9.0 - np.log10(values), np.nan)
    elif label_space != "raw":
        raise ValueError(label_space)
    return values.astype(float), label_space


def featurize_drugs(smiles: pd.Series, n_bits: int) -> sparse.csr_matrix:
    rows = []
    keep = []
    zero = np.zeros(n_bits, dtype=np.float32)
    for i, smi in enumerate(smiles.fillna("").astype(str)):
        fp = ecfp4(smi, n_bits=n_bits)
        rows.append(fp if fp is not None else zero)
        keep.append(i)
    return sparse.csr_matrix(np.vstack(rows).astype(np.float32))


def protein_kmer_features(seqs: pd.Series, n_bits: int, k: int) -> sparse.csr_matrix:
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for seq in seqs.fillna("").astype(str):
        counts: dict[int, int] = {}
        upper = seq.upper()
        for pos in range(0, max(len(upper) - k + 1, 0)):
            kmer = upper[pos : pos + k]
            if len(kmer) == k:
                idx = zlib.crc32(kmer.encode("ascii", errors="ignore")) % n_bits
                counts[idx] = counts.get(idx, 0) + 1
        norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
        for idx, count in sorted(counts.items()):
            indices.append(idx)
            data.append(float(count) / norm)
        indptr.append(len(indices))
    return sparse.csr_matrix((data, indices, indptr), shape=(len(seqs), n_bits), dtype=np.float32)


def featurize(frame: pd.DataFrame, drug_bits: int, protein_bits: int, kmer: int) -> sparse.csr_matrix:
    drug = featurize_drugs(frame["Drug"], drug_bits)
    protein = protein_kmer_features(frame["Target"], protein_bits, kmer)
    return sparse.hstack([drug, protein], format="csr")


def make_model(name: str, seed: int, n_jobs: int, n_estimators: int):
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=seed)
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "xgb":
        return XGBRegressor(
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
    if name == "lgbm":
        return LGBMRegressor(
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
    raise ValueError(name)


def sampled_cindex(y_true: np.ndarray, y_pred: np.ndarray, seed: int, max_pairs: int = 250_000) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = len(y_true)
    if n < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    if n <= 2500:
        i, j = np.triu_indices(n, k=1)
    else:
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
    diff = y_true[i] - y_true[j]
    informative = diff != 0
    if informative.sum() == 0:
        return np.nan
    pred_diff = y_pred[i] - y_pred[j]
    concordant = np.sign(diff[informative]) == np.sign(pred_diff[informative])
    tied = pred_diff[informative] == 0
    return float((concordant.sum() + 0.5 * tied.sum()) / informative.sum())


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, seed: int) -> dict[str, float]:
    out = regression_metrics(y_true, y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() >= 3:
        try:
            out["pearson"] = float(pearsonr(y_true[mask], y_pred[mask]).statistic)
        except Exception:
            out["pearson"] = np.nan
    else:
        out["pearson"] = np.nan
    out["cindex_sampled"] = sampled_cindex(y_true, y_pred, seed=seed)
    return out


def run_one(
    dataset: str,
    split: str,
    seed: int,
    model_name: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    parts = get_tdc_split(dataset, args.tdc_path, split, seed, args.frac)
    train = parts["train"].reset_index(drop=True)
    valid = parts["valid"].reset_index(drop=True)
    test = parts["test"].reset_index(drop=True)

    y_train, used_label_space = standardize_label(train["Y"], dataset, args.label_space)
    y_valid, _ = standardize_label(valid["Y"], dataset, used_label_space)
    y_test, _ = standardize_label(test["Y"], dataset, used_label_space)
    train_ok = np.isfinite(y_train)
    valid_ok = np.isfinite(y_valid)
    test_ok = np.isfinite(y_test)
    train = train.loc[train_ok].reset_index(drop=True)
    valid = valid.loc[valid_ok].reset_index(drop=True)
    test = test.loc[test_ok].reset_index(drop=True)
    y_train = y_train[train_ok]
    y_valid = y_valid[valid_ok]
    y_test = y_test[test_ok]

    combined = pd.concat([train, valid, test], ignore_index=True)
    x_all = featurize(combined, args.drug_bits, args.protein_bits, args.kmer)
    n_train = len(train)
    n_valid = len(valid)
    x_train = x_all[:n_train]
    x_valid = x_all[n_train : n_train + n_valid]
    x_test = x_all[n_train + n_valid :]

    y_mean = float(np.mean(y_train))
    y_std = max(float(np.std(y_train)), 1e-6)
    y_train_z = (y_train - y_mean) / y_std
    model = make_model(model_name, seed, args.n_jobs, args.n_estimators)
    model.fit(x_train, y_train_z)
    valid_pred = np.asarray(model.predict(x_valid), dtype=float) * y_std + y_mean
    test_pred = np.asarray(model.predict(x_test), dtype=float) * y_std + y_mean
    valid_metrics = evaluate(y_valid, valid_pred, seed)
    test_metrics = evaluate(y_test, test_pred, seed)

    train_drugs = set(train["Drug"].astype(str))
    test_drugs = set(test["Drug"].astype(str))
    train_targets = set(train["Target"].astype(str))
    test_targets = set(test["Target"].astype(str))
    row: dict[str, object] = {
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "model": model_name,
        "feature_set": f"ecfp{args.drug_bits}+protein{kmer_label(args.kmer, args.protein_bits)}",
        "label_space": used_label_space,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "test_rows": len(test),
        "train_test_drug_overlap": len(train_drugs & test_drugs),
        "train_test_target_overlap": len(train_targets & test_targets),
        "valid_mae": valid_metrics["mae"],
        "valid_rmse": valid_metrics["rmse"],
        "valid_spearman": valid_metrics["spearman"],
        "valid_pearson": valid_metrics["pearson"],
        "valid_cindex_sampled": valid_metrics["cindex_sampled"],
    }
    row.update({f"test_{key}": value for key, value in test_metrics.items()})
    return row


def prepare_case(dataset: str, split: str, seed: int, args: argparse.Namespace) -> dict[str, object]:
    parts = get_tdc_split(dataset, args.tdc_path, split, seed, args.frac)
    train = parts["train"].reset_index(drop=True)
    valid = parts["valid"].reset_index(drop=True)
    test = parts["test"].reset_index(drop=True)

    y_train, used_label_space = standardize_label(train["Y"], dataset, args.label_space)
    y_valid, _ = standardize_label(valid["Y"], dataset, used_label_space)
    y_test, _ = standardize_label(test["Y"], dataset, used_label_space)
    train_ok = np.isfinite(y_train)
    valid_ok = np.isfinite(y_valid)
    test_ok = np.isfinite(y_test)
    train = train.loc[train_ok].reset_index(drop=True)
    valid = valid.loc[valid_ok].reset_index(drop=True)
    test = test.loc[test_ok].reset_index(drop=True)
    y_train = y_train[train_ok]
    y_valid = y_valid[valid_ok]
    y_test = y_test[test_ok]

    combined = pd.concat([train, valid, test], ignore_index=True)
    x_all = featurize(combined, args.drug_bits, args.protein_bits, args.kmer)
    n_train = len(train)
    n_valid = len(valid)
    y_mean = float(np.mean(y_train))
    y_std = max(float(np.std(y_train)), 1e-6)
    return {
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "train": train,
        "valid": valid,
        "test": test,
        "x_train": x_all[:n_train],
        "x_valid": x_all[n_train : n_train + n_valid],
        "x_test": x_all[n_train + n_valid :],
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "y_train_z": (y_train - y_mean) / y_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "label_space": used_label_space,
    }


def run_prepared(prepared: dict[str, object], model_name: str, args: argparse.Namespace) -> dict[str, object]:
    dataset = str(prepared["dataset"])
    split = str(prepared["split"])
    seed = int(prepared["seed"])
    model = make_model(model_name, seed, args.n_jobs, args.n_estimators)
    model.fit(prepared["x_train"], prepared["y_train_z"])
    y_mean = float(prepared["y_mean"])
    y_std = float(prepared["y_std"])
    valid_pred = np.asarray(model.predict(prepared["x_valid"]), dtype=float) * y_std + y_mean
    test_pred = np.asarray(model.predict(prepared["x_test"]), dtype=float) * y_std + y_mean
    valid_metrics = evaluate(prepared["y_valid"], valid_pred, seed)
    test_metrics = evaluate(prepared["y_test"], test_pred, seed)
    train = prepared["train"]
    test = prepared["test"]
    train_drugs = set(train["Drug"].astype(str))
    test_drugs = set(test["Drug"].astype(str))
    train_targets = set(train["Target"].astype(str))
    test_targets = set(test["Target"].astype(str))
    row: dict[str, object] = {
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "model": model_name,
        "feature_set": f"ecfp{args.drug_bits}+protein{kmer_label(args.kmer, args.protein_bits)}",
        "label_space": prepared["label_space"],
        "train_rows": len(train),
        "valid_rows": len(prepared["valid"]),
        "test_rows": len(test),
        "train_test_drug_overlap": len(train_drugs & test_drugs),
        "train_test_target_overlap": len(train_targets & test_targets),
        "valid_mae": valid_metrics["mae"],
        "valid_rmse": valid_metrics["rmse"],
        "valid_spearman": valid_metrics["spearman"],
        "valid_pearson": valid_metrics["pearson"],
        "valid_cindex_sampled": valid_metrics["cindex_sampled"],
    }
    row.update({f"test_{key}": value for key, value in test_metrics.items()})
    return row


def kmer_label(kmer: int, protein_bits: int) -> str:
    return f"k{kmer}_{protein_bits}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["DAVIS", "KIBA", "BindingDB_Kd"])
    parser.add_argument("--splits", nargs="+", default=["random", "cold_drug", "cold_target", "cold_both"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2039, 2053])
    parser.add_argument("--models", nargs="+", default=["ridge", "xgb", "lgbm"], choices=["ridge", "rf", "xgb", "lgbm"])
    parser.add_argument("--tdc-path", type=Path, default=ROOT / "data" / "raw" / "tdc")
    parser.add_argument("--frac", nargs=3, type=float, default=[0.7, 0.1, 0.2])
    parser.add_argument("--label-space", choices=["auto", "raw", "pvalue"], default="auto")
    parser.add_argument("--drug-bits", type=int, default=2048)
    parser.add_argument("--protein-bits", type=int, default=2048)
    parser.add_argument("--kmer", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--tag", default="external_dti")
    args = parser.parse_args()

    args.n_jobs = min(args.n_jobs, cpu_threads())
    apply_thread_env(args.n_jobs)
    out_dir = ROOT / "results" / "tdc_dti"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets:
        for split in args.splits:
            for seed in args.seeds:
                print(f"[prepare] dataset={dataset} split={split} seed={seed}", flush=True)
                prepared = prepare_case(dataset, split, seed, args)
                for model_name in args.models:
                    print(f"[run] dataset={dataset} split={split} seed={seed} model={model_name}", flush=True)
                    row = run_prepared(prepared, model_name, args)
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(out_dir / f"tdc_dti_{args.tag}_partial.csv", index=False)
    out_csv = out_dir / f"tdc_dti_{args.tag}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    meta = {
        "datasets": args.datasets,
        "splits": args.splits,
        "seeds": args.seeds,
        "models": args.models,
        "frac": args.frac,
        "feature_set": f"ecfp{args.drug_bits}+protein{kmer_label(args.kmer, args.protein_bits)}",
    }
    (out_dir / f"tdc_dti_{args.tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] {out_csv}", flush=True)


if __name__ == "__main__":
    main()
