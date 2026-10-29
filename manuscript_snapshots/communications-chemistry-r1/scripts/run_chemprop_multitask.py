#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from canal.metrics import censored_direction_accuracy, censored_gaussian_nll, gaussian_nll, regression_metrics


def mixed_nll(frame: pd.DataFrame, pred: np.ndarray, sigma: float) -> float:
    exact = frame["p_relation"].eq("=").to_numpy()
    parts: list[float] = []
    weights: list[int] = []
    if exact.any():
        parts.append(gaussian_nll(frame.loc[exact, "p_activity"].to_numpy(), pred[exact], sigma))
        weights.append(int(exact.sum()))
    censored = ~exact
    if censored.any():
        parts.append(
            censored_gaussian_nll(
                frame.loc[censored, "p_activity"].to_numpy(),
                frame.loc[censored, "p_relation"].to_numpy(),
                pred[censored],
                sigma,
            )
        )
        weights.append(int(censored.sum()))
    return float(np.average(parts, weights=weights)) if parts else float("inf")


def calibrate_sigma(frame: pd.DataFrame, pred: np.ndarray) -> float:
    exact = frame["p_relation"].eq("=").to_numpy()
    fallback = float(np.std(frame.loc[exact, "p_activity"].to_numpy() - pred[exact])) if exact.sum() >= 20 else 1.0
    fallback = max(fallback, 0.05)
    grid = np.unique(
        np.concatenate([np.linspace(0.1, 1.5, 29), np.linspace(1.6, 3.0, 8), [fallback]])
    )
    return float(min(((mixed_nll(frame, pred, sigma), sigma) for sigma in grid), key=lambda item: item[0])[1])


def select_row_predictions(predictions: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    if len(predictions) != len(frame):
        raise RuntimeError(f"Prediction length mismatch: {len(predictions)} versus {len(frame)}")
    missing = sorted(set(frame["target_chembl_id"].astype(str)) - set(predictions.columns))
    if missing:
        raise RuntimeError(f"Missing prediction columns: {missing}")
    return np.array(
        [predictions.iloc[i][str(target)] for i, target in enumerate(frame["target_chembl_id"])],
        dtype=float,
    )


def evaluate_target(frame: pd.DataFrame, pred: np.ndarray, sigma: float) -> dict[str, float]:
    exact = frame["p_relation"].eq("=").to_numpy()
    censored = ~exact
    row: dict[str, float] = {"sigma": sigma, "test_n": int(exact.sum()), "test_censored_n": int(censored.sum())}
    if exact.any():
        metrics = regression_metrics(frame.loc[exact, "p_activity"].to_numpy(), pred[exact])
        row.update({f"test_{key}": value for key, value in metrics.items()})
    else:
        row.update({"test_mae": np.nan, "test_rmse": np.nan, "test_spearman": np.nan})
    if censored.any():
        row["test_censored_nll"] = censored_gaussian_nll(
            frame.loc[censored, "p_activity"].to_numpy(),
            frame.loc[censored, "p_relation"].to_numpy(),
            pred[censored],
            sigma,
        )
        row["test_censored_dir_acc"] = censored_direction_accuracy(
            frame.loc[censored, "p_activity"].to_numpy(),
            frame.loc[censored, "p_relation"].to_numpy(),
            pred[censored],
        )
    else:
        row["test_censored_nll"] = np.nan
        row["test_censored_dir_acc"] = np.nan
    assay = pd.DataFrame(
        {
            "assay": frame["assay_chembl_id"].to_numpy(),
            "exact": exact,
            "y": frame["p_activity"].to_numpy(dtype=float),
            "pred": pred,
        }
    )
    supported = (
        assay.query("exact")
        .groupby("assay")
        .apply(lambda group: np.mean(np.abs(group["pred"] - group["y"])) if len(group) >= 10 else np.nan, include_groups=False)
        .dropna()
    )
    row["worst_assay_mae"] = float(supported.max()) if len(supported) else np.nan
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["S1", "S2", "S3", "S4", "S5"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--treatment", choices=["exact_only", "censored_as_exact", "bounded_mse"], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tag", default="modern5")
    args = parser.parse_args()

    manifest = json.loads((args.prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    targets = [str(target) for target in manifest["targets"]]
    descriptor_columns = [str(column) for column in manifest["descriptor_columns"][args.split]]
    training_csv = args.prepared_dir / f"{args.split.lower()}_{args.treatment}.csv"
    val_csv = args.prepared_dir / f"{args.split.lower()}_val_smiles.csv"
    test_csv = args.prepared_dir / f"{args.split.lower()}_test_smiles.csv"
    run_name = f"chemprop_{args.split.lower()}_{args.treatment}_s{args.seed}_{args.tag}"
    model_dir = ROOT / "results" / "modern" / "chemprop_models" / run_name
    output_dir = ROOT / "results" / "modern" / "chemprop"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{run_name}.csv"
    if output_csv.exists() and output_csv.stat().st_size:
        print(f"[skip] {output_csv}")
        return
    model_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["OMP_NUM_THREADS"] = str(max(1, args.num_workers))
    env["MKL_NUM_THREADS"] = str(max(1, args.num_workers))
    env["TOKENIZERS_PARALLELISM"] = "false"
    loss = "bounded-mse" if args.treatment == "bounded_mse" else "mse"
    train_cmd = [
        str(Path(sys.executable).parent / "chemprop"),
        "train",
        "--data-path",
        str(training_csv),
        "--smiles-columns",
        "smiles",
        "--target-columns",
        *targets,
        "--descriptors-columns",
        *descriptor_columns,
        "--splits-column",
        "split",
        "--task-type",
        "regression",
        "--loss-function",
        loss,
        "--output-dir",
        str(model_dir),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--accelerator",
        "gpu",
        "--devices",
        "1",
        "--pytorch-seed",
        str(args.seed),
        "--data-seed",
        str(args.seed),
    ]
    models = sorted(model_dir.rglob("*.pt"))
    if not models:
        models = sorted(model_dir.rglob("*.ckpt"))
    if models:
        train_seconds = float("nan")
        print(f"[resume] using existing model {models[0]}")
    else:
        started = time.time()
        subprocess.run(train_cmd, check=True, env=env)
        train_seconds = time.time() - started
        models = sorted(model_dir.rglob("*.pt"))
        if not models:
            models = sorted(model_dir.rglob("*.ckpt"))
    if not models:
        raise RuntimeError(f"No trained model found under {model_dir}")
    model_path = models[0]
    prediction_paths: dict[str, Path] = {}
    for partition, input_csv in (("val", val_csv), ("test", test_csv)):
        pred_path = model_dir / f"{partition}_predictions.csv"
        if pred_path.exists() and pred_path.stat().st_size:
            print(f"[resume] using existing predictions {pred_path}")
            prediction_paths[partition] = pred_path
            continue
        pred_cmd = [
            str(Path(sys.executable).parent / "chemprop"),
            "predict",
            "--test-path",
            str(input_csv),
            "--model-paths",
            str(model_path),
            "--preds-path",
            str(pred_path),
            "--smiles-columns",
            "smiles",
            "--descriptors-columns",
            *descriptor_columns,
            "--accelerator",
            "gpu",
            "--devices",
            "1",
            "--num-workers",
            str(args.num_workers),
        ]
        subprocess.run(pred_cmd, check=True, env=env)
        prediction_paths[partition] = pred_path

    df = pd.read_parquet(args.dataset).reset_index(drop=True)
    split_col = f"split_{args.split.lower()}"
    val = df[df[split_col].eq("val")].reset_index(drop=True)
    test = df[df[split_col].eq("test")].reset_index(drop=True)
    val_predictions = pd.read_csv(prediction_paths["val"])
    test_predictions = pd.read_csv(prediction_paths["test"])
    val_pred = select_row_predictions(val_predictions, val)
    test_pred = select_row_predictions(test_predictions, test)

    rows: list[dict[str, object]] = []
    for target in targets:
        val_mask = val["target_chembl_id"].eq(target).to_numpy()
        test_mask = test["target_chembl_id"].eq(target).to_numpy()
        target_val = val.loc[val_mask].reset_index(drop=True)
        target_test = test.loc[test_mask].reset_index(drop=True)
        if len(target_test) < 30 or len(target_val) == 0:
            rows.append(
                {
                    "model": "chemprop_dmpnn",
                    "split": args.split,
                    "seed": args.seed,
                    "target": target,
                    "treatment": args.treatment,
                    "status": "too_few_rows",
                    "val_rows": len(target_val),
                    "test_rows": len(target_test),
                }
            )
            continue
        sigma = calibrate_sigma(target_val, val_pred[val_mask])
        row: dict[str, object] = {
            "model": "chemprop_dmpnn",
            "split": args.split,
            "seed": args.seed,
            "target": target,
            "treatment": args.treatment,
            "loss": loss,
            "status": "ok",
            "train_seconds": train_seconds,
            "val_rows": len(target_val),
            "test_rows": len(target_test),
            "gpu": args.gpu,
            "epochs_requested": args.epochs,
        }
        row.update(evaluate_target(target_test, test_pred[test_mask], sigma))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    provenance = {
        "training_command": train_cmd,
        "model_path": str(model_path),
        "training_seconds": train_seconds,
        "output": str(output_csv),
    }
    output_csv.with_suffix(".json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(output_csv)


if __name__ == "__main__":
    main()

