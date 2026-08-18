#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau
from tqdm import tqdm

from run_benchmark import (
    SPLITS_DEFAULT,
    SEEDS_DEFAULT,
    add_split,
    censor_arrays,
    ensure_dir,
    eval_predictions,
    fit_sigma,
    make_semisynthetic,
    split_leakage_audit,
    torch_censored_loss,
    write_json,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_vocab(smiles: pd.Series) -> dict[str, int]:
    chars = sorted({ch for smi in smiles.fillna("").astype(str) for ch in smi})
    return {ch: i + 2 for i, ch in enumerate(chars)} | {"<pad>": 0, "<unk>": 1}


def encode_smiles(smiles: pd.Series, vocab: dict[str, int], max_len: int) -> np.ndarray:
    arr = np.zeros((len(smiles), max_len), dtype=np.int16)
    unk = vocab["<unk>"]
    for i, smi in enumerate(tqdm(smiles.fillna("").astype(str), desc="encode_smiles")):
        toks = [vocab.get(ch, unk) for ch in smi[:max_len]]
        if toks:
            arr[i, : len(toks)] = np.asarray(toks, dtype=np.int16)
    return arr


class SmilesCNN(nn.Module):
    def __init__(self, vocab_size: int, n_targets: int, emb_dim: int = 64, channels: int = 128, target_dim: int = 64, dropout: float = 0.12):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, channels, k, padding=k // 2) for k in (3, 5, 7)])
        self.target_emb = nn.Embedding(n_targets, target_dim)
        in_dim = channels * len(self.convs) + target_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(128, 1)
        self.global_log_sigma = nn.Parameter(torch.tensor(-0.45))

    def forward(self, tokens: torch.Tensor, target_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.emb(tokens).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = F.silu(conv(x))
            pooled.append(torch.amax(h, dim=-1))
        target = self.target_emb(target_ids)
        h = torch.cat(pooled + [target], dim=1)
        h = self.mlp(h)
        mu = self.mu(h).squeeze(-1)
        log_sigma = self.global_log_sigma.expand_as(mu).clamp(math.log(0.12), math.log(3.0))
        return mu, log_sigma


def batch_iter(indices: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    idx = np.asarray(indices, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield idx[start : start + batch_size]


def tensor_batch(tokens: np.ndarray, target_ids: np.ndarray, y: np.ndarray, ctype: np.ndarray, idx: np.ndarray, device: str):
    xb = torch.as_tensor(tokens[idx].astype(np.int64), device=device)
    tb = torch.as_tensor(target_ids[idx].astype(np.int64), device=device)
    yb = torch.as_tensor(y[idx].astype(np.float32), device=device)
    cb = torch.as_tensor(ctype[idx].astype(np.int64), device=device)
    return xb, tb, yb, cb


def compute_loss(mode: str, mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor, ctype: torch.Tensor) -> torch.Tensor:
    if mode == "censored":
        return torch_censored_loss(mu, log_sigma, y, ctype)
    if mode == "exact":
        mask = ctype == 0
        if not mask.any():
            return torch.tensor(0.0, device=mu.device, requires_grad=True)
        return F.mse_loss(mu[mask], y[mask])
    if mode == "naive":
        return F.mse_loss(mu, y)
    raise KeyError(mode)


def eval_loss(model: nn.Module, tokens: np.ndarray, target_ids: np.ndarray, y: np.ndarray, ctype: np.ndarray, idx: np.ndarray, mode: str, device: str, batch_size: int) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for bidx in batch_iter(idx, batch_size, shuffle=False, seed=0):
            xb, tb, yb, cb = tensor_batch(tokens, target_ids, y, ctype, bidx, device)
            mu, log_sigma = model(xb, tb)
            loss = compute_loss(mode, mu, log_sigma, yb, cb)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_smiles_cnn(
    tokens: np.ndarray,
    target_ids: np.ndarray,
    y: np.ndarray,
    ctype: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    mode: str,
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    vocab_size: int,
    n_targets: int,
) -> nn.Module:
    set_seed(seed)
    model = SmilesCNN(vocab_size=vocab_size, n_targets=n_targets).to(device)
    with torch.no_grad():
        exact = idx_train[ctype[idx_train] == 0]
        init_idx = exact if len(exact) >= 20 else idx_train
        model.mu.bias.fill_(float(np.nanmean(y[init_idx])))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    best = float("inf")
    best_state = None
    stale = 0
    for epoch in range(epochs):
        model.train()
        for bidx in batch_iter(idx_train, batch_size, shuffle=True, seed=seed + epoch):
            xb, tb, yb, cb = tensor_batch(tokens, target_ids, y, ctype, bidx, device)
            opt.zero_grad(set_to_none=True)
            mu, log_sigma = model(xb, tb)
            loss = compute_loss(mode, mu, log_sigma, yb, cb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val = eval_loss(model, tokens, target_ids, y, ctype, idx_val, mode, device, batch_size)
        if val + 1e-4 < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model: nn.Module, tokens: np.ndarray, target_ids: np.ndarray, idx: np.ndarray, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    mus, sigmas = [], []
    dummy_y = np.zeros(len(tokens), dtype=np.float32)
    dummy_c = np.zeros(len(tokens), dtype=np.int64)
    with torch.no_grad():
        for bidx in batch_iter(idx, batch_size, shuffle=False, seed=0):
            xb, tb, _, _ = tensor_batch(tokens, target_ids, dummy_y, dummy_c, bidx, device)
            mu, log_sigma = model(xb, tb)
            mus.append(mu.detach().cpu().numpy())
            sigmas.append(torch.exp(log_sigma).detach().cpu().numpy())
    return np.concatenate(mus), np.concatenate(sigmas)


def aggregate_and_rank(out_dir: Path) -> None:
    files = sorted(p for p in out_dir.glob("metrics_*.csv") if p.name != "metrics_long.csv")
    if not files:
        return
    metrics = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    metrics.to_csv(out_dir / "metrics_long.csv", index=False)
    value_cols = [c for c in metrics.columns if c in {"mae_exact", "rmse_exact", "spearman_exact", "censored_nll", "boundary_satisfaction", "interval_miss_rate", "sigma", "latent_mae_all", "latent_rmse_all", "latent_spearman_all"}]
    if value_cols:
        agg = metrics.groupby(["semisynthetic", "split", "method"])[value_cols].agg(["mean", "std", "count"]).reset_index()
        agg.columns = ["_".join([str(x) for x in c if str(x)]).strip("_") if isinstance(c, tuple) else str(c) for c in agg.columns]
        agg.to_csv(out_dir / "metric_mean_std_by_split_method.csv", index=False)
        overall = metrics.groupby(["semisynthetic", "method"])[value_cols].agg(["mean", "std", "count"]).reset_index()
        overall.columns = ["_".join([str(x) for x in c if str(x)]).strip("_") if isinstance(c, tuple) else str(c) for c in overall.columns]
        overall.to_csv(out_dir / "metric_mean_std_overall_by_method.csv", index=False)
    rows = []
    for (semi, split, seed), g in metrics.dropna(subset=["mae_exact", "censored_nll"]).groupby(["semisynthetic", "split", "seed"]):
        if g["method"].nunique() < 2:
            continue
        rows.append(
            {
                "semisynthetic": bool(semi),
                "split": split,
                "seed": int(seed),
                "top_mae": str(g.loc[g["mae_exact"].idxmin(), "method"]),
                "top_nll": str(g.loc[g["censored_nll"].idxmin(), "method"]),
                "top1_changed": bool(g.loc[g["mae_exact"].idxmin(), "method"] != g.loc[g["censored_nll"].idxmin(), "method"]),
                "kendall_rank_corr": float(kendalltau(g["mae_exact"].rank(), g["censored_nll"].rank()).statistic),
            }
        )
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "ranking_distortion.csv", index=False)


def run_suite(df: pd.DataFrame, out_dir: Path, splits: list[str], seeds: list[int], modes: list[str], device: str, batch_size: int, epochs: int, patience: int, lr: float, max_len: int, semisynthetic: bool) -> pd.DataFrame:
    ensure_dir(out_dir)
    vocab = build_vocab(df["canonical_smiles_rdkit"])
    tokens = encode_smiles(df["canonical_smiles_rdkit"], vocab, max_len=max_len)
    target_codes, target_uniques = pd.factorize(df["target_endpoint"].astype(str), sort=True)
    target_ids = target_codes.astype(np.int64)
    y, ctype = censor_arrays(df)
    latent = df["latent_y_true"].to_numpy(dtype=np.float32) if "latent_y_true" in df.columns else None
    split_roles = {(split, seed): add_split(df, split, seed) for split in splits for seed in seeds}
    audit_dir = ensure_dir(out_dir / ("audits_semisynthetic" if semisynthetic else "audits"))
    split_leakage_audit(df, split_roles, audit_dir)
    all_rows = []
    for seed in seeds:
        for split in splits:
            roles = split_roles[(split, seed)]
            idx_train = np.flatnonzero(roles.eq("train").to_numpy())
            idx_val = np.flatnonzero(roles.eq("val").to_numpy())
            idx_test = np.flatnonzero(roles.eq("test").to_numpy())
            test_groups = df["target_endpoint"].fillna("__NA__").astype(str).to_numpy()[idx_test]
            if len(idx_train) < 100 or len(idx_val) < 30 or len(idx_test) < 30:
                continue
            print(f"[{'semi' if semisynthetic else 'real'}] split={split} seed={seed} train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}", flush=True)
            rows = []
            for mode in modes:
                method = f"smiles_cnn_{'tobit' if mode == 'censored' else mode}_molecule"
                train_idx = idx_train if mode != "exact" else idx_train[ctype[idx_train] == 0]
                if len(train_idx) < 50:
                    continue
                model = train_smiles_cnn(tokens, target_ids, y, ctype, train_idx, idx_val, mode, device, seed, epochs, batch_size, lr, patience, len(vocab), len(target_uniques))
                pred_val, sigma_val_pred = predict(model, tokens, target_ids, idx_val, device, batch_size)
                pred_test, sigma_test_pred = predict(model, tokens, target_ids, idx_test, device, batch_size)
                sigma = float(np.mean(sigma_test_pred)) if mode == "censored" else fit_sigma(y[idx_val], pred_val, ctype[idx_val])
                metrics = eval_predictions(
                    pred_test,
                    y[idx_test],
                    ctype[idx_test],
                    sigma=sigma,
                    latent_y=latent[idx_test] if latent is not None else None,
                    groups=test_groups,
                )
                row = {
                    "split": split,
                    "seed": seed,
                    "method": method,
                    "model_family": "smiles_cnn",
                    "feature_set": "smiles_target",
                    "censor_treatment": mode,
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(idx_val)),
                    "n_test": int(len(idx_test)),
                    "semisynthetic": bool(semisynthetic),
                    "method_set": "smiles_cnn",
                    **metrics,
                }
                rows.append(row)
            case = pd.DataFrame(rows)
            case.to_csv(out_dir / f"metrics_{'semi' if semisynthetic else 'real'}_{split}_seed{seed}.csv", index=False)
            roles.to_frame("role").to_csv(out_dir / f"split_roles_{split}_seed{seed}.csv", index_label="row_index")
            all_rows.extend(rows)
    return pd.DataFrame(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--artifacts-root", default="artifacts/smiles_cnn")
    parser.add_argument("--splits", nargs="+", default=SPLITS_DEFAULT)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    parser.add_argument("--modes", nargs="+", choices=["censored", "naive", "exact"], default=["censored"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--semisynthetic", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    root = ensure_dir(args.artifacts_root)
    df = pd.read_parquet(args.data)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=20260812).reset_index(drop=True)
    real_rows = run_suite(df, root, args.splits, args.seeds, args.modes, args.device, args.batch_size, args.epochs, args.patience, args.lr, args.max_len, semisynthetic=False)
    all_rows = [real_rows]
    if args.semisynthetic:
        semi = make_semisynthetic(df, seed=20260812)
        semi_rows = run_suite(semi, root, args.splits, args.seeds, args.modes, args.device, args.batch_size, args.epochs, args.patience, args.lr, args.max_len, semisynthetic=True)
        all_rows.append(semi_rows)
    aggregate_and_rank(root)
    manifest = {
        "data": args.data,
        "rows": int(len(df)),
        "target_endpoints": int(df["target_endpoint"].nunique()),
        "splits": args.splits,
        "seeds": args.seeds,
        "modes": args.modes,
        "device": args.device,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "max_len": args.max_len,
        "semisynthetic": bool(args.semisynthetic),
        "metric_rows": int(sum(len(x) for x in all_rows if x is not None)),
    }
    write_json(root / "suite_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
