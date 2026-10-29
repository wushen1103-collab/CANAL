#!/usr/bin/env python
"""Run the published UQ4DD MVE-Tobit molecular baseline on CANAL splits.

The network and optimizer follow MolecularAI/uq4dd commit
1da3d3d3c790e133b5f242218cde7d4eefa63814. The likelihood is evaluated
with log-CDF functions for numerical stability while preserving the same
Gaussian Tobit objective.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_ndtr
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from canal.resources import apply_thread_env, cpu_threads
from run_problem_check import featurize, filter_requested_panel, with_metadata_features


UQ4DD_COMMIT = "1da3d3d3c790e133b5f242218cde7d4eefa63814"


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise SystemExit("PyTorch is required for the UQ4DD baseline.") from exc
    return torch, nn, functional


def build_model(input_dim: int, torch, nn):
    class MVENetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            current = input_dim
            for _ in range(4):
                layers.extend([nn.Linear(current, 512), nn.ReLU(), nn.Dropout(0.5)])
                current = 512
            self.backbone = nn.Sequential(*layers)
            self.output = nn.Linear(512, 2)

        def forward(self, x):
            output = self.output(self.backbone(x))
            mean = output[:, 0]
            sigma = torch.nn.functional.softplus(output[:, 1]) + 1e-6
            return mean, sigma

    return MVENetwork()


def relation_codes(frame: pd.DataFrame) -> np.ndarray:
    return frame["p_relation"].map({"=": 0, "<": -1, ">": 1}).to_numpy(np.int64)


def to_dense_float(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def tobit_loss(mean, sigma, label, relation, torch):
    variance = sigma.square()
    z = (label - mean) / sigma
    exact = 0.5 * torch.log(2.0 * math.pi * variance) + 0.5 * z.square()
    left = -torch.special.log_ndtr(z)
    right = -torch.special.log_ndtr(-z)
    return torch.where(relation.eq(0), exact, torch.where(relation.lt(0), left, right)).mean()


def validation_loss(model, x, y, relation, device, batch_size, torch) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            stop = min(start + batch_size, len(y))
            xb = torch.from_numpy(x[start:stop]).to(device)
            yb = torch.from_numpy(y[start:stop]).to(device)
            rb = torch.from_numpy(relation[start:stop]).to(device)
            mean, sigma = model(xb)
            loss = tobit_loss(mean, sigma, yb, rb, torch)
            total += float(loss.detach().cpu()) * (stop - start)
            count += stop - start
    return total / max(count, 1)


def predict(model, x, device, batch_size, torch) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    means = []
    sigmas = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            mean, sigma = model(xb)
            means.append(mean.detach().cpu().numpy())
            sigmas.append(sigma.detach().cpu().numpy())
    return np.concatenate(means), np.concatenate(sigmas)


def evaluate(test: pd.DataFrame, mean: np.ndarray, sigma: np.ndarray) -> dict[str, object]:
    output: dict[str, object] = {}
    relation = test["p_relation"].astype(str).to_numpy()
    threshold = test["p_activity"].to_numpy(dtype=float)
    sigma = np.clip(np.asarray(sigma, dtype=float), 1e-6, None)
    exact = relation == "="
    if exact.any():
        y = threshold[exact]
        pred = mean[exact]
        error = pred - y
        output.update(
            {
                "test_n": int(exact.sum()),
                "test_mae": float(np.mean(np.abs(error))),
                "test_rmse": float(np.sqrt(np.mean(error**2))),
                "test_r2": float(r2_score(y, pred)) if len(y) >= 2 else np.nan,
                "test_spearman": float(spearmanr(y, pred).statistic) if len(y) >= 3 else np.nan,
                "test_kendall": float(kendalltau(y, pred).statistic) if len(y) >= 3 else np.nan,
                "test_nll": float(
                    np.mean(0.5 * np.log(2.0 * np.pi * sigma[exact] ** 2) + 0.5 * (error / sigma[exact]) ** 2)
                ),
            }
        )
    censored = ~exact
    if censored.any():
        z = (threshold[censored] - mean[censored]) / sigma[censored]
        row_nll = np.where(relation[censored] == "<", -log_ndtr(z), -log_ndtr(-z))
        correct = np.where(
            relation[censored] == ">",
            mean[censored] > threshold[censored],
            mean[censored] < threshold[censored],
        )
        violation = np.where(
            relation[censored] == ">",
            np.maximum(threshold[censored] - mean[censored], 0.0),
            np.maximum(mean[censored] - threshold[censored], 0.0),
        )
        output.update(
            {
                "test_censored_n": int(censored.sum()),
                "test_censored_nll": float(np.mean(row_nll)),
                "test_censored_nll_median": float(np.median(row_nll)),
                "test_censored_dir_acc": float(np.mean(correct)),
                "test_violation_distance": float(np.mean(violation)),
            }
        )
    assay = pd.DataFrame(
        {
            "assay": test.loc[exact, "assay_chembl_id"].astype(str).to_numpy(),
            "error": np.abs(mean[exact] - threshold[exact]),
        }
    )
    grouped = assay.groupby("assay")["error"].agg(["mean", "size"])
    grouped = grouped.loc[grouped["size"].ge(10)]
    if len(grouped):
        worst = str(grouped["mean"].idxmax())
        output["worst_assay_id"] = worst
        output["worst_assay_mae"] = float(grouped.loc[worst, "mean"])
        output["worst_assay_exact_n"] = int(grouped.loc[worst, "size"])
    else:
        output["worst_assay_id"] = None
        output["worst_assay_mae"] = np.nan
        output["worst_assay_exact_n"] = 0
    output["sigma_mean"] = float(np.mean(sigma))
    output["sigma_median"] = float(np.median(sigma))
    return output


def run_one(df, x_all, target: str, split: str, seed: int, args) -> dict[str, object] | None:
    torch, nn, _ = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    split_column = f"split_{split.lower()}"
    subset = df.loc[df["target_chembl_id"].eq(target)].copy()
    train = subset.loc[subset[split_column].eq("train")].copy()
    validation = subset.loc[subset[split_column].eq("val")].copy()
    test = subset.loc[subset[split_column].eq("test")].copy()
    if len(train) < 100 or len(validation) < 20 or len(test) < 30:
        return None

    x_train = x_all[train.index.to_numpy()]
    x_validation = x_all[validation.index.to_numpy()]
    x_test = x_all[test.index.to_numpy()]
    x_train, x_validation, x_test = with_metadata_features(
        train, validation, test, x_train, x_validation, x_test, args.metadata_mode
    )
    x_train = to_dense_float(x_train)
    x_validation = to_dense_float(x_validation)
    x_test = to_dense_float(x_test)
    y_train = train["p_activity"].to_numpy(np.float32)
    y_validation = validation["p_activity"].to_numpy(np.float32)
    relation_train = relation_codes(train)
    relation_validation = relation_codes(validation)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(x_train.shape[1], torch, nn).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.scheduler_patience
    )
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_state = None
    patience = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            xb = torch.from_numpy(x_train[index]).to(device)
            yb = torch.from_numpy(y_train[index]).to(device)
            rb = torch.from_numpy(relation_train[index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            mean, sigma = model(xb)
            loss = tobit_loss(mean, sigma, yb, rb, torch)
            loss.backward()
            optimizer.step()

        score = validation_loss(
            model,
            x_validation,
            y_validation,
            relation_validation,
            device,
            args.batch_size,
            torch,
        )
        scheduler.step(score)
        if score < best_loss - args.min_delta:
            best_loss = score
            best_state = copy.deepcopy(model.state_dict())
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mean, test_sigma = predict(model, x_test, device, args.batch_size, torch)
    row: dict[str, object] = {
        "endpoint": "IC50",
        "target": target,
        "split": split,
        "seed": seed,
        "treatment": "uq4dd_mve_tobit",
        "model": "UQ4DD-MVE",
        "status": "ok",
        "train_rows": len(train),
        "val_rows": len(validation),
        "test_rows": len(test),
        "epochs_ran": epoch,
        "best_val_tobit_nll": best_loss,
        "feature_set": args.feature_set,
        "n_bits": args.n_bits,
        "metadata_mode": args.metadata_mode,
        "uq4dd_commit": UQ4DD_COMMIT,
    }
    row.update(evaluate(test, test_mean, test_sigma))
    del model, x_train, x_validation, x_test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["S1", "S2", "S3", "S4"])
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--max-targets", type=int, default=20)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--feature-set", default="ecfp", choices=["ecfp", "ecfp_desc"])
    parser.add_argument("--n-bits", type=int, default=1024)
    parser.add_argument("--metadata-mode", default="none", choices=["none", "source_light"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--scheduler-patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()

    threads = min(args.cpu_threads, cpu_threads())
    apply_thread_env(threads)
    torch, _, _ = require_torch()
    torch.set_num_threads(threads)

    raw = pd.read_parquet(args.dataset).reset_index(drop=True)
    frame, targets = filter_requested_panel(raw, ["IC50"], args.max_targets)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    selected_targets = targets["IC50"][args.shard_index :: args.num_shards]
    targets["IC50"] = selected_targets
    frame = frame.loc[frame["target_chembl_id"].isin(selected_targets)].reset_index(drop=True)
    x_all = featurize(
        frame["canonical_smiles_rdkit"], n_bits=args.n_bits, feature_set=args.feature_set
    )
    rows = []
    partial = args.out.with_name(args.out.stem + "_partial.csv")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for target in targets["IC50"]:
        for split in args.splits:
            for seed in args.seeds:
                print(f"[run] target={target} split={split} seed={seed}", flush=True)
                try:
                    row = run_one(frame, x_all, target, split, seed, args)
                except Exception as exc:
                    row = {
                        "endpoint": "IC50",
                        "target": target,
                        "split": split,
                        "seed": seed,
                        "treatment": "uq4dd_mve_tobit",
                        "model": "UQ4DD-MVE",
                        "status": "error",
                        "error": repr(exc),
                    }
                if row is not None:
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(partial, index=False)

    result = pd.DataFrame(rows)
    result.to_csv(args.out, index=False)
    manifest = {
        "dataset": str(args.dataset),
        "rows": len(result),
        "ok_rows": int(result["status"].eq("ok").sum()) if len(result) else 0,
        "published_source": "Svensson et al., Artificial Intelligence in the Life Sciences 7 (2025) 100128",
        "source_repository": "https://github.com/MolecularAI/uq4dd",
        "source_commit": UQ4DD_COMMIT,
        "adaptation": "UQ4DD MVE architecture and Gaussian Tobit objective on CANAL IC50 splits; stable log-CDF evaluation",
        "configuration": vars(args),
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"[done] {args.out} rows={len(result)}", flush=True)


if __name__ == "__main__":
    main()
