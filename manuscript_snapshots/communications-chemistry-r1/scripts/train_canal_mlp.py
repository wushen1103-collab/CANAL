#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from canal.chem import ecfp4
from canal.metrics import censored_direction_accuracy, censored_gaussian_nll, gaussian_nll, interval_coverage, regression_metrics
from canal.resources import apply_thread_env, cpu_threads


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit("PyTorch is required for train_canal_mlp.py. Install torch in .conda first.") from exc
    return torch, nn, F


def featurize_dense(smiles: pd.Series, n_bits: int = 2048) -> np.ndarray:
    arrs = []
    for smi in smiles:
        fp = ecfp4(str(smi), n_bits=n_bits)
        if fp is None:
            raise ValueError(f"Bad SMILES during featurization: {smi}")
        arrs.append(fp)
    return np.vstack(arrs).astype(np.float32)


def encode_categories(train: pd.DataFrame, *parts: pd.DataFrame, column: str):
    cats = {v: i + 1 for i, v in enumerate(train[column].fillna("NA").astype(str).unique())}
    encoded = []
    for df in (train, *parts):
        encoded.append(df[column].fillna("NA").astype(str).map(cats).fillna(0).astype(np.int64).to_numpy())
    counts = np.ones(len(cats) + 1, dtype=np.float32)
    train_ids = encoded[0]
    for idx, n in pd.Series(train_ids).value_counts().items():
        counts[int(idx)] = float(n)
    counts[0] = 1.0
    return encoded, counts


@dataclass
class Arrays:
    x: np.ndarray
    y: np.ndarray
    rel_code: np.ndarray
    assay: np.ndarray
    doc: np.ndarray


def standardize_arrays(arrays: list[Arrays], mean: float, std: float) -> None:
    for arr in arrays:
        arr.y = ((arr.y - mean) / std).astype(np.float32)


def make_arrays(df: pd.DataFrame, assay_ids: np.ndarray, doc_ids: np.ndarray) -> Arrays:
    rel_map = {"=": 0, "<": -1, ">": 1}
    return Arrays(
        x=featurize_dense(df["canonical_smiles_rdkit"]),
        y=df["p_activity"].to_numpy(np.float32),
        rel_code=df["p_relation"].map(rel_map).fillna(0).astype(np.int64).to_numpy(),
        assay=assay_ids.astype(np.int64),
        doc=doc_ids.astype(np.int64),
    )


class CanalMLP:
    def __init__(self, n_assay: int, n_doc: int, assay_channel: str, torch, nn):
        super().__init__()
        self.torch = torch
        self.nn = nn
        self.assay_channel = assay_channel
        self.backbone = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        self.mu = nn.Linear(256, 1)
        self.global_log_sigma = nn.Parameter(torch.tensor(-0.25))
        if assay_channel in {"assay_bias", "assay_doc_bias"}:
            self.assay_bias = nn.Embedding(n_assay, 1, padding_idx=0)
            nn.init.zeros_(self.assay_bias.weight)
        else:
            self.assay_bias = None
        if assay_channel == "assay_doc_bias":
            self.doc_bias = nn.Embedding(n_doc, 1, padding_idx=0)
            nn.init.zeros_(self.doc_bias.weight)
        else:
            self.doc_bias = None

    def parameters(self):
        params = list(self.backbone.parameters()) + list(self.mu.parameters()) + [self.global_log_sigma]
        if self.assay_bias is not None:
            params += list(self.assay_bias.parameters())
        if self.doc_bias is not None:
            params += list(self.doc_bias.parameters())
        return params

    def train(self):
        self.backbone.train()
        self.mu.train()

    def eval(self):
        self.backbone.eval()
        self.mu.eval()

    def to(self, device):
        self.backbone.to(device)
        self.mu.to(device)
        self.global_log_sigma.data = self.global_log_sigma.data.to(device)
        if self.assay_bias is not None:
            self.assay_bias.to(device)
        if self.doc_bias is not None:
            self.doc_bias.to(device)
        return self

    def state_dict(self):
        state = {
            "backbone": {k: v.detach().cpu().clone() for k, v in self.backbone.state_dict().items()},
            "mu": {k: v.detach().cpu().clone() for k, v in self.mu.state_dict().items()},
            "global_log_sigma": self.global_log_sigma.detach().cpu().clone(),
        }
        if self.assay_bias is not None:
            state["assay_bias"] = {k: v.detach().cpu().clone() for k, v in self.assay_bias.state_dict().items()}
        if self.doc_bias is not None:
            state["doc_bias"] = {k: v.detach().cpu().clone() for k, v in self.doc_bias.state_dict().items()}
        return state

    def load_state_dict(self, state):
        self.backbone.load_state_dict(state["backbone"])
        self.mu.load_state_dict(state["mu"])
        self.global_log_sigma.data.copy_(state["global_log_sigma"].to(self.global_log_sigma.device))
        if self.assay_bias is not None and "assay_bias" in state:
            self.assay_bias.load_state_dict(state["assay_bias"])
        if self.doc_bias is not None and "doc_bias" in state:
            self.doc_bias.load_state_dict(state["doc_bias"])

    def __call__(self, x, assay, doc):
        h = self.backbone(x)
        pred = self.mu(h).squeeze(-1)
        if self.assay_bias is not None:
            pred = pred + self.assay_bias(assay).squeeze(-1)
        if self.doc_bias is not None:
            pred = pred + self.doc_bias(doc).squeeze(-1)
        sigma = self.torch.nn.functional.softplus(self.global_log_sigma) + 0.05
        return pred, sigma


def torch_loss(mu, sigma, y, rel, censor_weight, torch):
    z = (y - mu) / sigma
    cdf = torch.clamp(0.5 * (1.0 + torch.erf(z / math.sqrt(2.0))), 1e-8, 1.0 - 1e-8)
    exact = rel == 0
    left = rel == -1
    right = rel == 1
    losses = []
    if exact.any():
        err = (y[exact] - mu[exact]) / sigma
        losses.append(0.5 * torch.log(2 * torch.tensor(math.pi, device=y.device) * sigma * sigma) + 0.5 * err.pow(2).mean())
    if left.any():
        losses.append(censor_weight * (-torch.log(cdf[left])).mean())
    if right.any():
        losses.append(censor_weight * (-torch.log(1.0 - cdf[right])).mean())
    if not losses:
        return mu.mean() * 0.0
    return sum(losses)


def shrinkage_penalty(model, assay_counts, doc_counts, assay_l2, doc_l2, device, torch):
    penalty = torch.tensor(0.0, device=device)
    if model.assay_bias is not None and assay_l2 > 0:
        counts = torch.tensor(assay_counts, device=device)
        weights = 1.0 / torch.sqrt(torch.clamp(counts, min=1.0))
        penalty = penalty + assay_l2 * (weights[:, None] * model.assay_bias.weight.pow(2)).mean()
    if model.doc_bias is not None and doc_l2 > 0:
        counts = torch.tensor(doc_counts, device=device)
        weights = 1.0 / torch.sqrt(torch.clamp(counts, min=1.0))
        penalty = penalty + doc_l2 * (weights[:, None] * model.doc_bias.weight.pow(2)).mean()
    return penalty


def predict(model, arrays: Arrays, device, torch, batch_size: int = 4096):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(arrays.y), batch_size):
            sl = slice(start, start + batch_size)
            x = torch.tensor(arrays.x[sl], device=device)
            assay = torch.tensor(arrays.assay[sl], device=device)
            doc = torch.tensor(arrays.doc[sl], device=device)
            mu, sigma = model(x, assay, doc)
            preds.append(mu.detach().cpu().numpy())
    return np.concatenate(preds), float(sigma.detach().cpu())


def evaluate_frame(df_split: pd.DataFrame, pred: np.ndarray, sigma: float) -> dict[str, float]:
    exact = df_split["p_relation"].eq("=")
    row = {}
    if exact.any():
        row.update(regression_metrics(df_split.loc[exact, "p_activity"].to_numpy(), pred[exact.to_numpy()]))
        row["nll"] = gaussian_nll(df_split.loc[exact, "p_activity"].to_numpy(), pred[exact.to_numpy()], sigma)
        row["coverage90"] = interval_coverage(df_split.loc[exact, "p_activity"].to_numpy(), pred[exact.to_numpy()], sigma)
    cens = ~exact
    row["censored_n"] = int(cens.sum())
    row["censored_nll"] = censored_gaussian_nll(
        df_split.loc[cens, "p_activity"].to_numpy(),
        df_split.loc[cens, "p_relation"].to_numpy(),
        pred[cens.to_numpy()],
        sigma,
    )
    row["censored_dir_acc"] = censored_direction_accuracy(
        df_split.loc[cens, "p_activity"].to_numpy(),
        df_split.loc[cens, "p_relation"].to_numpy(),
        pred[cens.to_numpy()],
    )
    return row


def run_one(df: pd.DataFrame, endpoint: str, target: str, split: str, seed: int, args) -> dict[str, object] | None:
    torch, nn, _ = require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)

    split_col = f"split_{split.lower()}"
    sub = df[(df["endpoint"] == endpoint) & (df["target_chembl_id"] == target)].copy()
    train = sub[sub[split_col] == "train"].copy()
    val = sub[sub[split_col] == "val"].copy()
    test = sub[sub[split_col] == "test"].copy()
    if args.treatment == "exact_only":
        train = train[train["p_relation"] == "="].copy()
    if len(train) < 100 or len(test) < 30:
        return None

    (assay_train, assay_val, assay_test), assay_counts = encode_categories(train, val, test, column="assay_chembl_id")
    (doc_train, doc_val, doc_test), doc_counts = encode_categories(train, val, test, column="document_chembl_id")
    train_a = make_arrays(train, assay_train, doc_train)
    val_a = make_arrays(val, assay_val, doc_val)
    test_a = make_arrays(test, assay_test, doc_test)
    exact_train = train["p_relation"].eq("=")
    scale_values = train.loc[exact_train, "p_activity"].to_numpy(np.float32) if exact_train.any() else train["p_activity"].to_numpy(np.float32)
    y_mean = float(np.mean(scale_values))
    y_std = max(float(np.std(scale_values)), 0.25)
    standardize_arrays([train_a, val_a, test_a], y_mean, y_std)
    val_score_df = val.copy()
    val_score_df["p_activity"] = val_a.y

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CanalMLP(len(assay_counts), len(doc_counts), args.assay_channel, torch, nn).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = len(train_a.y)
    rng = np.random.default_rng(seed)

    exact_indices = np.flatnonzero(train_a.rel_code == 0)
    if args.pretrain_epochs > 0 and len(exact_indices) > 0:
        for _ in range(args.pretrain_epochs):
            model.train()
            order = rng.permutation(exact_indices)
            for start in range(0, len(order), args.batch_size):
                idx = order[start : start + args.batch_size]
                x = torch.tensor(train_a.x[idx], device=device)
                y = torch.tensor(train_a.y[idx], device=device)
                rel = torch.zeros(len(idx), dtype=torch.long, device=device)
                assay = torch.tensor(train_a.assay[idx], device=device)
                doc = torch.tensor(train_a.doc[idx], device=device)
                opt.zero_grad(set_to_none=True)
                mu, sigma = model(x, assay, doc)
                loss = torch_loss(mu, sigma, y, rel, 0.0, torch)
                loss = loss + shrinkage_penalty(model, assay_counts, doc_counts, args.assay_l2, args.doc_l2, device, torch)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

    best_state = None
    best_val = float("inf")
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(n)
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            x = torch.tensor(train_a.x[idx], device=device)
            y = torch.tensor(train_a.y[idx], device=device)
            rel = torch.tensor(train_a.rel_code[idx], device=device)
            assay = torch.tensor(train_a.assay[idx], device=device)
            doc = torch.tensor(train_a.doc[idx], device=device)
            opt.zero_grad(set_to_none=True)
            mu, sigma = model(x, assay, doc)
            loss = torch_loss(mu, sigma, y, rel, args.censor_weight, torch)
            loss = loss + shrinkage_penalty(model, assay_counts, doc_counts, args.assay_l2, args.doc_l2, device, torch)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        val_pred, val_sigma = predict(model, val_a, device, torch, batch_size=args.batch_size)
        val_metrics = evaluate_frame(val_score_df, val_pred, val_sigma)
        score_parts = []
        if np.isfinite(val_metrics.get("censored_nll", np.nan)):
            score_parts.append(val_metrics["censored_nll"])
        exact_val = val["p_relation"].eq("=")
        if exact_val.any():
            exact_nll = gaussian_nll(val_score_df.loc[exact_val, "p_activity"].to_numpy(), val_pred[exact_val.to_numpy()], val_sigma)
            if np.isfinite(exact_nll):
                score_parts.append(exact_nll)
        val_score = float(np.mean(score_parts)) if score_parts else float("inf")
        if np.isfinite(val_score) and val_score < best_val:
            best_val = val_score
            best_state = model.state_dict()
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_pred_norm, test_sigma_norm = predict(model, test_a, device, torch, batch_size=args.batch_size)
    test_pred = test_pred_norm * y_std + y_mean
    test_sigma = test_sigma_norm * y_std
    metrics = evaluate_frame(test, test_pred, test_sigma)
    row = {
        "endpoint": endpoint,
        "target": target,
        "split": split,
        "seed": seed,
        "treatment": args.treatment,
        "assay_channel": args.assay_channel,
        "censor_weight": args.censor_weight,
        "epochs_ran": epoch,
        "pretrain_epochs": args.pretrain_epochs,
        "device": str(device),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "sigma": test_sigma,
        "y_mean": y_mean,
        "y_std": y_std,
        "best_val": best_val,
    }
    row.update({f"test_{k}": v for k, v in metrics.items()})
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--endpoints", nargs="+", default=["IC50"])
    parser.add_argument("--splits", nargs="+", default=["S1", "S2", "S3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027])
    parser.add_argument("--max-targets", type=int, default=3)
    parser.add_argument("--treatment", choices=["exact_only", "censored_likelihood"], default="censored_likelihood")
    parser.add_argument("--assay-channel", choices=["none", "assay_bias", "assay_doc_bias"], default="assay_doc_bias")
    parser.add_argument("--censor-weight", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--assay-l2", type=float, default=1e-3)
    parser.add_argument("--doc-l2", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--tag", default="smoke")
    args = parser.parse_args()

    threads = min(args.cpu_threads, cpu_threads())
    apply_thread_env(threads)
    df = pd.read_parquet(args.dataset).reset_index(drop=True)
    rows = []
    for endpoint in args.endpoints:
        targets = df[df["endpoint"] == endpoint].groupby("target_chembl_id").size().sort_values(ascending=False).head(args.max_targets).index.tolist()
        for target in targets:
            for split in args.splits:
                for seed in args.seeds:
                    print(f"[run] endpoint={endpoint} target={target} split={split} seed={seed} treatment={args.treatment} channel={args.assay_channel}", flush=True)
                    row = run_one(df, endpoint, target, split, seed, args)
                    if row:
                        rows.append(row)
                        out_dir = ROOT / "results" / "canal_mlp"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        pd.DataFrame(rows).to_csv(out_dir / f"canal_mlp_{args.tag}_partial.csv", index=False)
    out_dir = ROOT / "results" / "canal_mlp"
    out_path = out_dir / f"canal_mlp_{args.tag}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    (out_dir / f"canal_mlp_{args.tag}.json").write_text(json.dumps({"rows": len(rows), "tag": args.tag}, indent=2), encoding="utf-8")
    print(f"[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
