#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_chemprop_multitask import calibrate_sigma, evaluate_target


class GraphCache:
    def __init__(self, path: Path) -> None:
        self.nodes = np.load(path / "node_features.npy", mmap_mode="r")
        self.edges = np.load(path / "edge_index.npy", mmap_mode="r")
        self.edge_features = np.load(path / "edge_features.npy", mmap_mode="r")
        self.node_ptr = np.load(path / "node_ptr.npy", mmap_mode="r")
        self.edge_ptr = np.load(path / "edge_ptr.npy", mmap_mode="r")
        self.row_graph_index = np.load(path / "row_graph_index.npy", mmap_mode="r")


class ActivityDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_dir: Path,
        row_indices: np.ndarray,
        source: np.ndarray,
        task_index: np.ndarray,
        target: np.ndarray,
        lt_mask: np.ndarray,
        gt_mask: np.ndarray,
    ) -> None:
        self.cache_dir = cache_dir
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.source = np.asarray(source, dtype=np.float32)
        self.task_index = np.asarray(task_index, dtype=np.int64)
        self.target = np.asarray(target, dtype=np.float32)
        self.lt_mask = np.asarray(lt_mask, dtype=bool)
        self.gt_mask = np.asarray(gt_mask, dtype=bool)
        self.cache: GraphCache | None = None

    def __len__(self) -> int:
        return len(self.row_indices)

    def _cache(self) -> GraphCache:
        if self.cache is None:
            self.cache = GraphCache(self.cache_dir)
        return self.cache

    def __getitem__(self, item: int) -> Data:
        cache = self._cache()
        row = int(self.row_indices[item])
        graph = int(cache.row_graph_index[row])
        node_start, node_end = int(cache.node_ptr[graph]), int(cache.node_ptr[graph + 1])
        edge_start, edge_end = int(cache.edge_ptr[graph]), int(cache.edge_ptr[graph + 1])
        return Data(
            x=torch.as_tensor(np.asarray(cache.nodes[node_start:node_end], dtype=np.float32)),
            edge_index=torch.as_tensor(np.asarray(cache.edges[:, edge_start:edge_end], dtype=np.int64)),
            edge_attr=torch.as_tensor(np.asarray(cache.edge_features[edge_start:edge_end], dtype=np.float32)),
            source=torch.from_numpy(self.source[row]).view(1, -1),
            task_id=torch.tensor([self.task_index[row]], dtype=torch.long),
            y=torch.tensor([self.target[row]], dtype=torch.float32),
            lt_mask=torch.tensor([bool(self.lt_mask[row])], dtype=torch.bool),
            gt_mask=torch.tensor([bool(self.gt_mask[row])], dtype=torch.bool),
            row_id=torch.tensor([row], dtype=torch.long),
        )


class SourceAttentiveFP(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, source_dim: int, tasks: int) -> None:
        super().__init__()
        self.encoder = AttentiveFP(
            in_channels=node_dim,
            hidden_channels=200,
            out_channels=200,
            edge_dim=edge_dim,
            num_layers=2,
            num_timesteps=2,
            dropout=0.2,
        )
        self.head = nn.Sequential(
            nn.Linear(200 + source_dim, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, tasks),
        )

    def forward(self, batch: Data) -> torch.Tensor:
        molecular = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return self.head(torch.cat([molecular, batch.source], dim=1))


def selected_predictions(model: nn.Module, batch: Data) -> torch.Tensor:
    output = model(batch)
    return output.gather(1, batch.task_id.view(-1, 1)).squeeze(1)


def bounded_mse(pred: torch.Tensor, batch: Data, treatment: str) -> torch.Tensor:
    error = pred - batch.y.view(-1)
    if treatment == "bounded_mse":
        satisfied = (batch.lt_mask.view(-1) & pred.lt(batch.y.view(-1))) | (
            batch.gt_mask.view(-1) & pred.gt(batch.y.view(-1))
        )
        error = torch.where(satisfied, torch.zeros_like(error), error)
    return error.square().mean()


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, target_mean: np.ndarray, target_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    rows: list[np.ndarray] = []
    values: list[np.ndarray] = []
    means = torch.as_tensor(target_mean, device=device, dtype=torch.float32)
    stds = torch.as_tensor(target_std, device=device, dtype=torch.float32)
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            normalized = selected_predictions(model, batch)
        task = batch.task_id.view(-1)
        denormalized = normalized.float() * stds[task] + means[task]
        rows.append(batch.row_id.view(-1).cpu().numpy())
        values.append(denormalized.cpu().numpy())
    return np.concatenate(rows), np.concatenate(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--graph-cache", type=Path, required=True)
    parser.add_argument("--split", choices=["S1", "S2", "S3", "S4", "S5"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--treatment", choices=["exact_only", "censored_as_exact", "bounded_mse"], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tag", default="modern_src5")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(f"cuda:{args.gpu}")

    output_dir = ROOT / "results" / "modern" / "attentivefp"
    model_dir = ROOT / "results" / "modern" / "attentivefp_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"attentivefp_{args.split.lower()}_{args.treatment}_s{args.seed}_{args.tag}"
    output_csv = output_dir / f"{run_name}.csv"
    checkpoint = model_dir / f"{run_name}.pt"
    if output_csv.exists() and output_csv.stat().st_size:
        print(f"[skip] {output_csv}")
        return

    manifest = json.loads((args.prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    graph_manifest = json.loads((args.graph_cache / "manifest.json").read_text(encoding="utf-8"))
    targets = [str(target) for target in manifest["targets"]]
    descriptor_columns = [str(column) for column in manifest["descriptor_columns"][args.split]]
    df = pd.read_parquet(args.dataset).reset_index(drop=True)
    source_frame = pd.read_csv(
        args.prepared_dir / f"{args.split.lower()}_exact_only.csv",
        usecols=["split", *descriptor_columns],
    )
    if len(df) != len(source_frame) or len(df) != int(graph_manifest["rows"]):
        raise RuntimeError("Dataset, source descriptor, and graph-cache row counts differ")
    partitions = source_frame["split"].astype(str).to_numpy()
    source = source_frame[descriptor_columns].to_numpy(dtype=np.float32)
    target_lookup = {target: index for index, target in enumerate(targets)}
    task_index = df["target_chembl_id"].astype(str).map(target_lookup).to_numpy(dtype=np.int64)
    target = df["p_activity"].to_numpy(dtype=np.float32)
    relation = df["p_relation"].astype(str).to_numpy()
    lt_mask = relation == "<" if args.treatment == "bounded_mse" else np.zeros(len(df), dtype=bool)
    gt_mask = relation == ">" if args.treatment == "bounded_mse" else np.zeros(len(df), dtype=bool)
    include = relation == "=" if args.treatment == "exact_only" else np.ones(len(df), dtype=bool)

    train_indices = np.flatnonzero((partitions == "train") & include)
    val_indices = np.flatnonzero((partitions == "val") & include)
    val_eval_indices = np.flatnonzero(partitions == "val")
    test_indices = np.flatnonzero(partitions == "test")
    target_mean = np.zeros(len(targets), dtype=np.float32)
    target_std = np.ones(len(targets), dtype=np.float32)
    for task in range(len(targets)):
        values = target[train_indices[task_index[train_indices] == task]]
        target_mean[task] = float(values.mean())
        target_std[task] = max(float(values.std()), 0.1)
    normalized_target = (target - target_mean[task_index]) / target_std[task_index]
    source_mean = source[train_indices].mean(axis=0)
    source_std = source[train_indices].std(axis=0)
    source_std[source_std < 1e-6] = 1.0
    source = (source - source_mean) / source_std

    dataset_args = (args.graph_cache, source, task_index, normalized_target, lt_mask, gt_mask)
    train_dataset = ActivityDataset(dataset_args[0], train_indices, *dataset_args[1:])
    val_dataset = ActivityDataset(dataset_args[0], val_indices, *dataset_args[1:])
    val_eval_dataset = ActivityDataset(dataset_args[0], val_eval_indices, *dataset_args[1:])
    test_dataset = ActivityDataset(dataset_args[0], test_indices, *dataset_args[1:])
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    val_eval_loader = DataLoader(val_eval_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = SourceAttentiveFP(
        int(graph_manifest["node_feature_dim"]),
        int(graph_manifest["edge_feature_dim"]),
        source.shape[1],
        len(targets),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    started = time.time()
    history: list[dict[str, float | int]] = []
    for epoch in range(args.epochs):
        model.train()
        train_sum = 0.0
        train_n = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = bounded_mse(selected_predictions(model, batch), batch, args.treatment)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_sum += float(loss.detach()) * batch.num_graphs
            train_n += batch.num_graphs
        model.eval()
        val_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = bounded_mse(selected_predictions(model, batch), batch, args.treatment)
                val_sum += float(loss) * batch.num_graphs
                val_n += batch.num_graphs
        train_loss = train_sum / max(train_n, 1)
        val_loss = val_sum / max(val_n, 1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch + 1} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch + 1
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    train_seconds = time.time() - started
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    val_rows, val_values = predict(model, val_eval_loader, device, target_mean, target_std)
    test_rows, test_values = predict(model, test_loader, device, target_mean, target_std)
    val_prediction = np.full(len(df), np.nan, dtype=float)
    test_prediction = np.full(len(df), np.nan, dtype=float)
    val_prediction[val_rows] = val_values
    test_prediction[test_rows] = test_values

    rows: list[dict[str, object]] = []
    for task, target_name in enumerate(targets):
        val_mask = (partitions == "val") & (task_index == task)
        test_mask = (partitions == "test") & (task_index == task)
        target_val = df.loc[val_mask].reset_index(drop=True)
        target_test = df.loc[test_mask].reset_index(drop=True)
        sigma = calibrate_sigma(target_val, val_prediction[val_mask])
        row: dict[str, object] = {
            "model": "attentivefp",
            "split": args.split,
            "seed": args.seed,
            "target": target_name,
            "treatment": args.treatment,
            "loss": "bounded_mse" if args.treatment == "bounded_mse" else "mse",
            "status": "ok",
            "train_seconds": train_seconds,
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "train_rows": len(train_indices),
            "val_rows": int(val_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "gpu": args.gpu,
            "epochs_requested": args.epochs,
        }
        row.update(evaluate_target(target_test, test_prediction[test_mask], sigma))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    provenance = {
        "model": "PyG AttentiveFP with source-light late fusion",
        "architecture": {"hidden_channels": 200, "num_layers": 2, "num_timesteps": 2, "dropout": 0.2},
        "optimizer": {"name": "Adam", "learning_rate": 1e-3, "weight_decay": 1e-6},
        "normalization": "training-only per-target labels and source columns",
        "checkpoint": str(checkpoint),
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "train_seconds": train_seconds,
        "history": history,
        "versions": {name: version(name) for name in ["torch", "torch-geometric", "rdkit", "pandas"]},
    }
    output_csv.with_suffix(".json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(output_csv)


if __name__ == "__main__":
    main()

