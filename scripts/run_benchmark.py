#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from scipy.special import log_ndtr, ndtr
from scipy.optimize import minimize
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None


RDLogger.DisableLog("rdApp.*")
RDLogger.logger().setLevel(RDLogger.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning)
_MORGAN_GENERATORS: dict[int, object] = {}


SEEDS_DEFAULT = [13, 29, 47, 61, 83]
SPLITS_DEFAULT = ["random", "scaffold", "source", "temporal", "joint"]

PROSPECTIVE_CATS = [
    "target_endpoint",
    "endpoint",
    "assay_type",
    "assay_test_type",
    "assay_category",
    "assay_organism",
    "bao_format",
]
RETROSPECTIVE_CATS = [
    "activity_src_id",
    "assay_src_id",
    "doc_type",
    "data_validity_comment",
    "activity_comment_present",
    "data_validity_comment_present",
    "doi_present",
    "pubmed_present",
]
QC_NUMS = ["confidence_score", "potential_duplicate", "publication_year"]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_hash(text: object, bins: int) -> int:
    h = hashlib.blake2b(str(text).encode("utf-8", errors="ignore"), digest_size=8).hexdigest()
    return int(h, 16) % bins


def mol_from_smiles(smiles: str):
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def fingerprint_and_desc(smiles: str, n_bits: int) -> tuple[np.ndarray, np.ndarray]:
    mol = mol_from_smiles(smiles)
    fp = np.zeros(n_bits, dtype=np.float32)
    desc = np.zeros(10, dtype=np.float32)
    if mol is None:
        return fp, desc
    generator = _MORGAN_GENERATORS.get(n_bits)
    if generator is None:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        _MORGAN_GENERATORS[n_bits] = generator
    bitvect = generator.GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(bitvect, fp)
    try:
        desc[:] = [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            mol.GetNumHeavyAtoms(),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Descriptors.FpDensityMorgan2(mol),
        ]
    except Exception:
        pass
    return fp, desc


def build_or_load_mol_features(df: pd.DataFrame, cache_dir: Path, n_bits: int, workers: int) -> tuple[np.ndarray, np.ndarray]:
    ensure_dir(cache_dir)
    key = hashlib.sha256(("|".join(df["canonical_smiles_rdkit"].astype(str).tolist()) + f"|{n_bits}").encode("utf-8")).hexdigest()[:16]
    fp_path = cache_dir / f"mol_fp_desc_{key}_{n_bits}.npz"
    if fp_path.exists():
        dat = np.load(fp_path)
        return dat["fp"].astype(np.float32), dat["desc"].astype(np.float32)
    rows = Parallel(n_jobs=max(1, workers), batch_size=128)(
        delayed(fingerprint_and_desc)(smi, n_bits) for smi in tqdm(df["canonical_smiles_rdkit"].astype(str), desc="molecular_features")
    )
    fp = np.vstack([r[0] for r in rows]).astype(np.float32)
    desc = np.vstack([r[1] for r in rows]).astype(np.float32)
    # Keep descriptor scale sane before concatenating with bits.
    med = np.nanmedian(desc, axis=0)
    desc = np.where(np.isfinite(desc), desc, med)
    np.savez_compressed(fp_path, fp=fp, desc=desc)
    return fp, desc


def hashed_categorical_frame(df: pd.DataFrame, cols: list[str], bins_per_col: int = 16) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    out = np.zeros((len(df), bins_per_col * len(cols)), dtype=np.float32)
    for j, col in enumerate(cols):
        vals = df[col].fillna("__NA__").astype(str) if col in df.columns else pd.Series(["__NA__"] * len(df))
        base = j * bins_per_col
        for i, val in enumerate(vals):
            out[i, base + stable_hash(val, bins_per_col)] = 1.0
    return out


def one_hot_categorical_frame(df: pd.DataFrame, col: str) -> tuple[np.ndarray, list[str]]:
    """Encode a low-cardinality task identity without hash collisions."""
    vals = df[col].fillna("__NA__").astype(str)
    categories = sorted(vals.unique().tolist())
    lookup = {value: idx for idx, value in enumerate(categories)}
    out = np.zeros((len(df), len(categories)), dtype=np.float32)
    for row_idx, value in enumerate(vals):
        out[row_idx, lookup[value]] = 1.0
    return out, categories


def numeric_frame(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    arrs = []
    for col in cols:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
        else:
            x = np.zeros(len(df), dtype=np.float32)
        if col == "publication_year":
            x = (x - np.nanmedian(x)) / max(1.0, np.nanstd(x))
        x = np.where(np.isfinite(x), x, np.nanmedian(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0)
        arrs.append(x.reshape(-1, 1))
    return np.hstack(arrs).astype(np.float32)


@dataclass
class FeatureStore:
    mol: np.ndarray
    desc: np.ndarray
    target: np.ndarray
    prospective: np.ndarray
    retrospective: np.ndarray
    qc: np.ndarray

    def matrix(self, name: str) -> np.ndarray:
        if name == "molecule_only":
            return np.hstack([self.mol, self.desc, self.target]).astype(np.float32)
        if name == "metadata_only":
            return np.hstack([self.target, self.prospective, self.retrospective, self.qc]).astype(np.float32)
        if name == "prospective":
            return np.hstack([self.mol, self.desc, self.target, self.prospective]).astype(np.float32)
        if name == "all_source_light":
            return np.hstack([self.mol, self.desc, self.target, self.prospective, self.retrospective, self.qc]).astype(np.float32)
        if name == "source_only":
            return np.hstack([self.target, self.prospective, self.retrospective, self.qc]).astype(np.float32)
        raise KeyError(name)


def build_features(df: pd.DataFrame, cache_dir: Path, n_bits: int, workers: int, permute_metadata_seed: int | None = None) -> FeatureStore:
    mol, desc = build_or_load_mol_features(df, cache_dir, n_bits, workers)
    meta = df.copy()
    for src_col, dst_col in [
        ("activity_comment", "activity_comment_present"),
        ("data_validity_comment", "data_validity_comment_present"),
        ("doi", "doi_present"),
        ("pubmed_id", "pubmed_present"),
    ]:
        if src_col in meta.columns:
            meta[dst_col] = meta[src_col].notna() & meta[src_col].astype(str).str.strip().ne("")
        else:
            meta[dst_col] = False
    if permute_metadata_seed is not None:
        rng = np.random.default_rng(permute_metadata_seed)
        # Keep target_endpoint fixed: it is the supervised task identifier, not a source/QC
        # metadata field. The permutation control should test assay/source/prospective
        # context rather than destroying the task definition.
        for col in [c for c in PROSPECTIVE_CATS if c != "target_endpoint"] + RETROSPECTIVE_CATS + QC_NUMS:
            if col in meta.columns:
                meta[col] = rng.permutation(meta[col].to_numpy())
    target, _ = one_hot_categorical_frame(meta, "target_endpoint")
    prospective = hashed_categorical_frame(meta, [c for c in PROSPECTIVE_CATS if c != "target_endpoint"], bins_per_col=16)
    retrospective = hashed_categorical_frame(meta, RETROSPECTIVE_CATS, bins_per_col=16)
    qc = numeric_frame(meta, QC_NUMS)
    return FeatureStore(mol=mol, desc=desc, target=target, prospective=prospective, retrospective=retrospective, qc=qc)


def add_split(df: pd.DataFrame, split: str, seed: int) -> pd.Series:
    roles = pd.Series("train", index=df.index, dtype=object)
    if split == "random":
        train_idx, temp_idx = train_test_split(df.index.to_numpy(), test_size=0.30, random_state=seed, stratify=df["target_endpoint"])
        val_idx, test_idx = train_test_split(temp_idx, test_size=2 / 3, random_state=seed + 1)
        roles.loc[val_idx] = "val"
        roles.loc[test_idx] = "test"
        return roles
    if split in {"scaffold", "source"}:
        group_col = "bemis_murcko_scaffold" if split == "scaffold" else "assay_chembl_id"
        groups = df[group_col].fillna("__NA__").astype(str)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_val, test = next(gss.split(df, groups=groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed + 1)
        train_rel, val_rel = next(gss2.split(df.iloc[train_val], groups=groups.iloc[train_val]))
        roles.iloc[train_val[val_rel]] = "val"
        roles.iloc[test] = "test"
        return roles
    if split == "temporal":
        order = df.sort_values(["publication_year", "activity_id"]).index.to_numpy()
        n = len(order)
        roles.loc[order[int(n * 0.70) : int(n * 0.80)]] = "val"
        roles.loc[order[int(n * 0.80) :]] = "test"
        return roles
    if split == "joint":
        # Joint OOD: latest records whose scaffolds were not observed in the early train period.
        year_cut = int(np.quantile(df["publication_year"], 0.75))
        early = df.index[df["publication_year"] < year_cut]
        late = df.index[df["publication_year"] >= year_cut]
        train_scaffolds = set(df.loc[early, "bemis_murcko_scaffold"].astype(str))
        candidate_test = df.loc[late].index[~df.loc[late, "bemis_murcko_scaffold"].astype(str).isin(train_scaffolds)]
        if len(candidate_test) < max(100, int(0.05 * len(df))):
            candidate_test = late
        rng = np.random.default_rng(seed)
        test_n = min(len(candidate_test), max(100, int(0.20 * len(df))))
        test_idx = rng.choice(candidate_test, size=test_n, replace=False)
        remaining = df.index.difference(test_idx)
        val_candidates = remaining[df.loc[remaining, "publication_year"] >= year_cut]
        if len(val_candidates) < max(50, int(0.05 * len(df))):
            val_candidates = remaining
        val_n = min(len(val_candidates), max(50, int(0.10 * len(df))))
        val_idx = rng.choice(val_candidates, size=val_n, replace=False)
        roles.loc[val_idx] = "val"
        roles.loc[test_idx] = "test"
        return roles
    raise ValueError(split)


def censor_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = pd.to_numeric(df["y_observed"], errors="coerce").to_numpy(dtype=np.float32)
    ctype = df["censor_type"].map({"exact": 0, "upper": 1, "lower": 2}).fillna(0).to_numpy(dtype=np.int64)
    return y, ctype


def exact_mask(ctype: np.ndarray) -> np.ndarray:
    return ctype == 0


def gaussian_censored_nll(mu: np.ndarray, sigma: float | np.ndarray, y: np.ndarray, ctype: np.ndarray) -> float:
    sigma_arr = np.asarray(sigma, dtype=float) + 1e-6
    z = (y - mu) / sigma_arr
    log_sigma = np.log(sigma_arr)
    exact = ctype == 0
    upper = ctype == 1
    lower = ctype == 2
    losses = []
    if exact.any():
        losses.append(0.5 * np.square(z[exact]) + log_sigma if np.ndim(log_sigma) == 0 else 0.5 * np.square(z[exact]) + log_sigma[exact])
    if upper.any():
        losses.append(-log_ndtr(z[upper]))
    if lower.any():
        losses.append(-log_ndtr(-z[lower]))
    if not losses:
        return float("nan")
    return float(np.mean(np.concatenate([np.asarray(x).reshape(-1) for x in losses])))


def boundary_metrics(mu: np.ndarray, sigma: float, y: np.ndarray, ctype: np.ndarray, latent_y: np.ndarray | None = None) -> dict:
    out: dict[str, float] = {}
    cens = ctype != 0
    if cens.any():
        sat = np.zeros_like(cens, dtype=bool)
        sat[ctype == 1] = mu[ctype == 1] <= y[ctype == 1]
        sat[ctype == 2] = mu[ctype == 2] >= y[ctype == 2]
        out["boundary_satisfaction"] = float(sat[cens].mean())
        z = (y[cens] - mu[cens]) / (sigma + 1e-6)
        probs = np.zeros(cens.sum(), dtype=float)
        csub = ctype[cens]
        probs[csub == 1] = ndtr(z[csub == 1])
        probs[csub == 2] = ndtr(-z[csub == 2])
        out["boundary_prob_mean"] = float(np.mean(probs))
        out["boundary_prob_ece_proxy"] = float(abs(np.mean(probs) - 1.0))
    else:
        out["boundary_satisfaction"] = np.nan
        out["boundary_prob_mean"] = np.nan
        out["boundary_prob_ece_proxy"] = np.nan
    if latent_y is not None:
        event = np.zeros_like(cens, dtype=bool)
        event[ctype == 1] = latent_y[ctype == 1] <= y[ctype == 1]
        event[ctype == 2] = latent_y[ctype == 2] >= y[ctype == 2]
        if cens.any():
            z = (y[cens] - mu[cens]) / (sigma + 1e-6)
            probs = np.zeros(cens.sum(), dtype=float)
            csub = ctype[cens]
            probs[csub == 1] = ndtr(z[csub == 1])
            probs[csub == 2] = ndtr(-z[csub == 2])
            truth = event[cens].astype(float)
            out["boundary_brier"] = float(np.mean((probs - truth) ** 2))
            out["boundary_ece"] = expected_calibration_error(probs, truth)
    return out


def expected_calibration_error(probs: np.ndarray, truth: np.ndarray, bins: int = 10) -> float:
    if len(probs) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(probs[mask].mean() - truth[mask].mean())
    return float(ece)


def macro_spearman(y: np.ndarray, mu: np.ndarray, groups: np.ndarray) -> tuple[float, int]:
    """Compute Spearman within each task and macro-average across evaluable tasks."""
    scores = []
    for group in np.unique(groups):
        mask = groups == group
        if mask.sum() <= 3 or np.unique(y[mask]).size < 2 or np.unique(mu[mask]).size < 2:
            continue
        score = spearmanr(y[mask], mu[mask]).correlation
        if np.isfinite(score):
            scores.append(float(score))
    return (float(np.mean(scores)), len(scores)) if scores else (float("nan"), 0)


def eval_predictions(
    mu: np.ndarray,
    y: np.ndarray,
    ctype: np.ndarray,
    sigma: float | None = None,
    latent_y: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> dict:
    exact = exact_mask(ctype)
    target_y = latent_y if latent_y is not None else y
    if sigma is None:
        if exact.any():
            sigma = float(np.std(target_y[exact] - mu[exact]) + 1e-3)
        else:
            sigma = float(np.std(target_y - mu) + 1e-3)
    out: dict[str, float] = {"sigma": float(np.mean(sigma))}
    if exact.any():
        out["mae_exact"] = float(mean_absolute_error(target_y[exact], mu[exact]))
        out["rmse_exact"] = float(math.sqrt(mean_squared_error(target_y[exact], mu[exact])))
        out["spearman_exact_pooled"] = float(spearmanr(target_y[exact], mu[exact]).correlation) if exact.sum() > 3 else np.nan
        if groups is not None:
            out["spearman_exact"], out["spearman_exact_task_count"] = macro_spearman(
                target_y[exact], mu[exact], np.asarray(groups)[exact]
            )
        else:
            out["spearman_exact"] = out["spearman_exact_pooled"]
            out["spearman_exact_task_count"] = np.nan
    else:
        out["mae_exact"] = np.nan
        out["rmse_exact"] = np.nan
        out["spearman_exact"] = np.nan
        out["spearman_exact_pooled"] = np.nan
        out["spearman_exact_task_count"] = 0
    if latent_y is not None:
        out["latent_mae_all"] = float(mean_absolute_error(latent_y, mu))
        out["latent_rmse_all"] = float(math.sqrt(mean_squared_error(latent_y, mu)))
        out["latent_spearman_all_pooled"] = float(spearmanr(latent_y, mu).correlation) if len(mu) > 3 else np.nan
        if groups is not None:
            out["latent_spearman_all"], out["latent_spearman_task_count"] = macro_spearman(
                latent_y, mu, np.asarray(groups)
            )
        else:
            out["latent_spearman_all"] = out["latent_spearman_all_pooled"]
            out["latent_spearman_task_count"] = np.nan
    out["censored_nll"] = gaussian_censored_nll(mu, sigma, y, ctype)
    out.update(boundary_metrics(mu, float(np.mean(sigma)), y, ctype, latent_y=latent_y))
    return out


def fit_sigma(y_val: np.ndarray, pred_val: np.ndarray, ctype_val: np.ndarray) -> float:
    m = exact_mask(ctype_val)
    if m.sum() >= 10:
        return float(np.clip(np.std(y_val[m] - pred_val[m]), 0.15, 3.0))
    return float(np.clip(np.std(y_val - pred_val), 0.15, 3.0))


def fit_censored_affine_calibration(y_val: np.ndarray, pred_val: np.ndarray, ctype_val: np.ndarray) -> tuple[float, float, float, float]:
    init_sigma = fit_sigma(y_val, pred_val, ctype_val)

    def objective(theta: np.ndarray) -> float:
        scale, bias, log_sigma = theta
        mu = scale * pred_val + bias
        sigma = float(np.exp(log_sigma))
        val = gaussian_censored_nll(mu, sigma, y_val, ctype_val)
        if not np.isfinite(val):
            return 1e9
        # Tiny shrinkage prevents wild affine transforms on small OOD validation folds.
        return float(val + 1e-4 * (scale - 1.0) ** 2 + 1e-4 * bias**2)

    identity_nll = objective(np.array([1.0, 0.0, math.log(init_sigma)], dtype=float))
    res = minimize(
        objective,
        x0=np.array([1.0, 0.0, math.log(init_sigma)], dtype=float),
        method="L-BFGS-B",
        bounds=[(0.25, 2.25), (-2.5, 2.5), (math.log(0.12), math.log(3.0))],
        options={"maxiter": 100, "ftol": 1e-7},
    )
    if not res.success or not np.isfinite(res.fun) or float(res.fun) > identity_nll + 1e-5:
        return 1.0, 0.0, init_sigma, identity_nll
    scale, bias, log_sigma = [float(x) for x in res.x]
    return scale, bias, float(np.exp(log_sigma)), float(res.fun)


def train_sklearn_model(method: str, x_train: np.ndarray, y_train: np.ndarray, workers: int, seed: int, tree_preset: str = "standard"):
    fast = tree_preset == "fast"
    if method == "ridge":
        model = Ridge(alpha=5.0, random_state=seed)
    elif method == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=120 if fast else 220,
            learning_rate=0.055 if fast else 0.045,
            l2_regularization=0.05,
            max_leaf_nodes=31,
            random_state=seed,
        )
    elif method == "rf":
        model = RandomForestRegressor(
            n_estimators=72 if fast else 160,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=max(1, workers),
            random_state=seed,
        )
    elif method == "extratrees":
        model = ExtraTreesRegressor(
            n_estimators=96 if fast else 180,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=max(1, workers),
            random_state=seed,
        )
    else:
        raise KeyError(method)
    model.fit(x_train, y_train)
    return model


class TorchCensoredNet(nn.Module):
    def __init__(self, dim: int, hidden: int = 384, dropout: float = 0.08, hetero: bool = True):
        super().__init__()
        self.hetero = hetero
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden // 2, 1)
        self.log_sigma = nn.Linear(hidden // 2, 1) if hetero else None
        self.global_log_sigma = nn.Parameter(torch.tensor(-0.5))

    def forward(self, x):
        h = self.net(x)
        mu = self.mu(h).squeeze(-1)
        if self.hetero:
            log_sigma = self.log_sigma(h).squeeze(-1).clamp(-3.0, 1.5)
        else:
            log_sigma = self.global_log_sigma.expand_as(mu).clamp(-3.0, 1.5)
        return mu, log_sigma


def torch_censored_loss(mu, log_sigma, y, ctype):
    sigma = torch.exp(log_sigma) + 1e-6
    z = (y - mu) / sigma
    exact = ctype == 0
    upper = ctype == 1
    lower = ctype == 2
    losses = []
    if exact.any():
        losses.append(0.5 * z[exact].pow(2) + log_sigma[exact])
    if upper.any():
        losses.append(-torch.special.log_ndtr(z[upper]))
    if lower.any():
        losses.append(-torch.special.log_ndtr(-z[lower]))
    if not losses:
        return torch.tensor(0.0, device=mu.device, requires_grad=True)
    return torch.cat([x.reshape(-1) for x in losses]).mean()


def train_torch_censored(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    c_val: np.ndarray,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    hidden: int,
    lr: float,
    hetero: bool,
):
    if torch is None:
        raise RuntimeError("torch is not available")
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_train_s = scaler.fit_transform(x_train).astype(np.float32)
    x_val_s = scaler.transform(x_val).astype(np.float32)
    model = TorchCensoredNet(x_train_s.shape[1], hidden=hidden, hetero=hetero).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best = float("inf")
    patience = 12
    stale = 0
    n = len(x_train_s)
    for epoch in range(epochs):
        order = np.random.permutation(n)
        model.train()
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = torch.from_numpy(x_train_s[idx]).to(device)
            yb = torch.from_numpy(y_train[idx].astype(np.float32)).to(device)
            cb = torch.from_numpy(c_train[idx].astype(np.int64)).to(device)
            mu, log_sigma = model(xb)
            loss = torch_censored_loss(mu, log_sigma, yb, cb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(x_val_s).to(device)
            yb = torch.from_numpy(y_val.astype(np.float32)).to(device)
            cb = torch.from_numpy(c_val.astype(np.int64)).to(device)
            mu, log_sigma = model(xb)
            val = float(torch_censored_loss(mu, log_sigma, yb, cb).detach().cpu())
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
    return model, scaler, best


def torch_predict(model, scaler: StandardScaler, x: np.ndarray, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    xs = scaler.transform(x).astype(np.float32)
    mus = []
    sigmas = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(xs), batch_size):
            xb = torch.from_numpy(xs[start : start + batch_size]).to(device)
            mu, log_sigma = model(xb)
            mus.append(mu.detach().cpu().numpy())
            sigmas.append(np.exp(log_sigma.detach().cpu().numpy()))
    return np.concatenate(mus), np.concatenate(sigmas)


def make_semisynthetic(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    exact_df = df[df["censor_type"].eq("exact")].copy()
    if len(exact_df) < 1000:
        raise RuntimeError("Semi-synthetic censoring needs at least 1000 exact rows.")
    rng = np.random.default_rng(seed)
    out = exact_df.copy().reset_index(drop=True)
    out["latent_y_true"] = out["y_observed"].astype(float)
    by_group = df.groupby("target_endpoint")
    censor_rate = by_group["is_censored"].mean().to_dict()
    rel_mix = by_group["censor_type"].value_counts(normalize=True).unstack(fill_value=0.0)
    threshold_pool = df[df["censor_type"].ne("exact")].groupby(["target_endpoint", "censor_type"])["p_activity_threshold"].apply(list).to_dict()
    new_y = out["latent_y_true"].to_numpy(copy=True)
    new_type = np.array(["exact"] * len(out), dtype=object)
    new_rel = np.array(["="] * len(out), dtype=object)
    for i, row in out.iterrows():
        g = row["target_endpoint"]
        p = float(censor_rate.get(g, 0.25))
        if rng.random() > p:
            continue
        lower_p = float(rel_mix.loc[g, "lower"]) if g in rel_mix.index and "lower" in rel_mix.columns else 0.35
        ctype = "lower" if rng.random() < lower_p else "upper"
        pool = threshold_pool.get((g, ctype)) or threshold_pool.get((g, "upper")) or threshold_pool.get((g, "lower"))
        if pool:
            thr = float(rng.choice(pool))
        else:
            noise = abs(float(rng.normal(0.35, 0.15)))
            thr = float(row["latent_y_true"] - noise if ctype == "lower" else row["latent_y_true"] + noise)
        # Ensure the synthetic inequality is true for the latent value.
        if ctype == "lower" and not (row["latent_y_true"] >= thr):
            thr = float(row["latent_y_true"] - abs(rng.normal(0.25, 0.1)))
        if ctype == "upper" and not (row["latent_y_true"] <= thr):
            thr = float(row["latent_y_true"] + abs(rng.normal(0.25, 0.1)))
        new_y[i] = thr
        new_type[i] = ctype
        new_rel[i] = "<" if ctype == "lower" else ">"
    out["y_observed"] = new_y
    out["p_activity_threshold"] = new_y
    out["censor_type"] = new_type
    out["standard_relation"] = new_rel
    out["is_censored"] = out["censor_type"].ne("exact")
    return out


def audit_tables(df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    summary = {
        "rows": int(len(df)),
        "target_endpoints": int(df["target_endpoint"].nunique()),
        "targets": int(df["target_chembl_id"].nunique()),
        "molecules": int(df["molecule_key"].nunique()),
        "scaffolds": int(df["bemis_murcko_scaffold"].nunique()),
        "assays": int(df["assay_chembl_id"].nunique()),
        "docs": int(df["document_chembl_id"].nunique()),
        "censor_rate": float(df["is_censored"].mean()),
        "exact_rate": float(1.0 - df["is_censored"].mean()),
        "relation_counts": df["standard_relation"].value_counts(dropna=False).to_dict(),
        "endpoint_counts": df["endpoint"].value_counts(dropna=False).to_dict(),
    }
    write_json(out_dir / "dataset_summary.json", summary)
    task_ids = sorted(df["target_endpoint"].fillna("__NA__").astype(str).unique().tolist())
    legacy_bins = [stable_hash(task_id, 32) for task_id in task_ids]
    bin_counts = pd.Series(legacy_bins).value_counts().to_dict()
    pd.DataFrame(
        {
            "target_endpoint": task_ids,
            "one_hot_index": np.arange(len(task_ids), dtype=int),
            "legacy_hash_bin": legacy_bins,
            "legacy_hash_collision": [bin_counts[b] > 1 for b in legacy_bins],
        }
    ).to_csv(out_dir / "target_identity_encoding_audit.csv", index=False)
    df.groupby(["endpoint", "target_endpoint", "standard_relation"]).size().rename("n").reset_index().to_csv(out_dir / "censor_relation_by_target.csv", index=False)
    mech_cols = ["target_endpoint", "endpoint", "assay_type", "assay_organism", "bao_format", "publication_year", "confidence_score", "activity_src_id", "assay_src_id"]
    rows = []
    for col in mech_cols:
        if col not in df.columns:
            continue
        g = df.groupby(col, dropna=False)["is_censored"].agg(["count", "mean"]).reset_index()
        g = g[g["count"] >= 20].sort_values("mean", ascending=False).head(100)
        g.insert(0, "factor", col)
        g.rename(columns={col: "level", "count": "n", "mean": "censor_rate"}, inplace=True)
        rows.append(g)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out_dir / "censoring_mechanism_top_levels.csv", index=False)
    rep = (
        df[df["censor_type"].eq("exact")]
        .groupby(["molecule_key", "target_endpoint"])["y_observed"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    rep["range"] = rep["max"] - rep["min"]
    rep = rep[rep["count"] >= 2].sort_values(["range", "count"], ascending=False)
    rep.to_csv(out_dir / "repeated_measurement_disagreement.csv", index=False)


def split_leakage_audit(df: pd.DataFrame, split_roles: dict[tuple[str, int], pd.Series], out_dir: Path) -> None:
    rows = []
    for (split, seed), roles in split_roles.items():
        train = df[roles.eq("train")]
        test = df[roles.eq("test")]
        row = {"split": split, "seed": seed, "train_n": len(train), "test_n": len(test)}
        for col, name in [
            ("molecule_key", "molecule"),
            ("bemis_murcko_scaffold", "scaffold"),
            ("assay_chembl_id", "assay"),
            ("document_chembl_id", "document"),
            ("target_endpoint", "target_endpoint"),
        ]:
            a = set(train[col].dropna().astype(str))
            b = set(test[col].dropna().astype(str))
            row[f"{name}_test_overlap_rate"] = float(len(a & b) / max(1, len(b)))
            row[f"{name}_overlap_n"] = int(len(a & b))
        dup_cols = ["molecule_key", "target_endpoint", "assay_chembl_id"]
        a = set(map(tuple, train[dup_cols].astype(str).to_numpy()))
        b = set(map(tuple, test[dup_cols].astype(str).to_numpy()))
        row["duplicate_pair_overlap_n"] = int(len(a & b))
        row["duplicate_pair_test_overlap_rate"] = float(len(a & b) / max(1, len(b)))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "split_leakage_audit.csv", index=False)


def run_one_case(
    df: pd.DataFrame,
    features: FeatureStore,
    split: str,
    seed: int,
    out_dir: Path,
    workers: int,
    device: str,
    quick: bool,
    semisynthetic: bool,
    method_set: str,
    tree_preset: str,
) -> pd.DataFrame:
    roles = add_split(df, split, seed)
    y, ctype = censor_arrays(df)
    latent = df["latent_y_true"].to_numpy(dtype=np.float32) if "latent_y_true" in df.columns else None
    idx_train = np.flatnonzero(roles.eq("train").to_numpy())
    idx_val = np.flatnonzero(roles.eq("val").to_numpy())
    idx_test = np.flatnonzero(roles.eq("test").to_numpy())
    test_groups = df["target_endpoint"].fillna("__NA__").astype(str).to_numpy()[idx_test]
    if len(idx_train) < 100 or len(idx_val) < 30 or len(idx_test) < 30:
        return pd.DataFrame()

    case_rows = []
    methods = [
        ("ridge", "molecule_only", "exact"),
        ("ridge", "molecule_only", "naive"),
        ("hgb", "molecule_only", "exact"),
        ("hgb", "molecule_only", "naive"),
        ("rf", "molecule_only", "naive"),
        ("extratrees", "molecule_only", "naive"),
        ("hgb", "metadata_only", "naive"),
        ("hgb", "prospective", "naive"),
        ("hgb", "all_source_light", "naive"),
    ]
    if method_set == "trees":
        methods = [m for m in methods if m[0] in {"hgb", "rf", "extratrees"}]
    elif method_set == "ridge":
        methods = [m for m in methods if m[0] == "ridge"]
    elif method_set == "core":
        methods = [m for m in methods if m[0] == "ridge"]
    elif method_set == "calibrated":
        methods = []
    elif quick:
        methods = [m for m in methods if m[0] == "ridge"]
    for model_name, feature_name, treatment in methods:
        x = features.matrix(feature_name)
        if treatment == "exact":
            train_mask = idx_train[ctype[idx_train] == 0]
        else:
            train_mask = idx_train
        if len(train_mask) < 50:
            continue
        model = train_sklearn_model(model_name, x[train_mask], y[train_mask], workers=workers, seed=seed, tree_preset=tree_preset)
        pred_val = model.predict(x[idx_val]).astype(float)
        pred_test = model.predict(x[idx_test]).astype(float)
        sigma = fit_sigma(y[idx_val], pred_val, ctype[idx_val])
        metrics = eval_predictions(
            pred_test,
            y[idx_test],
            ctype[idx_test],
            sigma=sigma,
            latent_y=latent[idx_test] if latent is not None else None,
            groups=test_groups,
        )
        case_rows.append(
            {
                "split": split,
                "seed": seed,
                "method": f"{model_name}_{feature_name}_{treatment}",
                "model_family": model_name,
                "feature_set": feature_name,
                "censor_treatment": treatment,
                "n_train": int(len(train_mask)),
                "n_val": int(len(idx_val)),
                "n_test": int(len(idx_test)),
                "semisynthetic": bool(semisynthetic),
                "method_set": method_set,
                "tree_preset": tree_preset if model_name in {"hgb", "rf", "extratrees"} else "",
                **metrics,
            }
        )

    if method_set == "calibrated":
        calibrated_methods = [
            ("canal_extratrees_molecule_calibrated", "extratrees", "molecule_only", "naive"),
            ("canal_hgb_source_calibrated", "hgb", "all_source_light", "naive"),
        ]
        for method_name, model_name, feature_name, treatment in calibrated_methods:
            x = features.matrix(feature_name)
            train_mask = idx_train if treatment == "naive" else idx_train[ctype[idx_train] == 0]
            if len(train_mask) < 50:
                continue
            model = train_sklearn_model(model_name, x[train_mask], y[train_mask], workers=workers, seed=seed, tree_preset=tree_preset)
            raw_val = model.predict(x[idx_val]).astype(float)
            raw_test = model.predict(x[idx_test]).astype(float)
            scale, bias, sigma, val_calibrated_nll = fit_censored_affine_calibration(y[idx_val], raw_val, ctype[idx_val])
            pred_test = scale * raw_test + bias
            metrics = eval_predictions(
                pred_test,
                y[idx_test],
                ctype[idx_test],
                sigma=sigma,
                latent_y=latent[idx_test] if latent is not None else None,
                groups=test_groups,
            )
            case_rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "method": method_name,
                    "model_family": model_name,
                    "feature_set": feature_name,
                    "censor_treatment": "naive_plus_censored_affine_calibration",
                    "n_train": int(len(train_mask)),
                    "n_val": int(len(idx_val)),
                    "n_test": int(len(idx_test)),
                    "semisynthetic": bool(semisynthetic),
                    "method_set": method_set,
                    "tree_preset": tree_preset,
                    "calibration_scale": scale,
                    "calibration_bias": bias,
                    "validation_censored_nll_calibrated": val_calibrated_nll,
                    **metrics,
                }
            )

    if method_set not in {"trees", "calibrated"} and torch is not None:
        torch_methods = [
            ("tobit_mlp_molecule", "molecule_only", False),
            ("gaussian_censored_mlp_molecule_hetero", "molecule_only", True),
            ("canal_source_mlp", "all_source_light", False),
            ("canal_source_mlp_hetero_sigma", "all_source_light", True),
            ("canal_prospective_mlp", "prospective", True),
        ]
        if quick:
            torch_methods = [torch_methods[0], torch_methods[2], torch_methods[-1]]
        for name, feature_name, hetero in torch_methods:
            x = features.matrix(feature_name)
            try:
                model, scaler, best = train_torch_censored(
                    x[idx_train],
                    y[idx_train],
                    ctype[idx_train],
                    x[idx_val],
                    y[idx_val],
                    ctype[idx_val],
                    seed=seed,
                    device=device,
                    epochs=45 if quick else 90,
                    batch_size=2048 if not quick else 4096,
                    hidden=256 if quick else 384,
                    lr=8e-4,
                    hetero=hetero,
                )
                mu_test, sig_test = torch_predict(model, scaler, x[idx_test], device=device, batch_size=8192)
                metrics = eval_predictions(
                    mu_test,
                    y[idx_test],
                    ctype[idx_test],
                    sigma=float(np.mean(sig_test)),
                    latent_y=latent[idx_test] if latent is not None else None,
                    groups=test_groups,
                )
                case_rows.append(
                    {
                        "split": split,
                        "seed": seed,
                        "method": name,
                        "model_family": "torch_mlp",
                        "feature_set": feature_name,
                        "censor_treatment": "censored_likelihood",
                        "n_train": int(len(idx_train)),
                        "n_val": int(len(idx_val)),
                        "n_test": int(len(idx_test)),
                        "val_censored_nll": float(best),
                        "semisynthetic": bool(semisynthetic),
                        "method_set": method_set,
                        **metrics,
                    }
                )
            except Exception as e:
                case_rows.append(
                    {
                        "split": split,
                        "seed": seed,
                        "method": name,
                        "model_family": "torch_mlp",
                        "feature_set": feature_name,
                        "censor_treatment": "censored_likelihood",
                        "error": repr(e),
                        "semisynthetic": bool(semisynthetic),
                        "method_set": method_set,
                    }
                )
    out = pd.DataFrame(case_rows)
    ensure_dir(out_dir)
    out.to_csv(out_dir / f"metrics_{'semi' if semisynthetic else 'real'}_{split}_seed{seed}.csv", index=False)
    roles.to_frame("role").to_csv(out_dir / f"split_roles_{split}_seed{seed}.csv", index_label="row_index")
    return out


def summarize_metrics(metrics: pd.DataFrame, out_dir: Path) -> None:
    if metrics.empty:
        return
    metrics.to_csv(out_dir / "metrics_long.csv", index=False)
    value_cols = [
        c
        for c in metrics.columns
        if c
        not in {
            "split",
            "seed",
            "method",
            "model_family",
            "feature_set",
            "censor_treatment",
            "error",
            "semisynthetic",
            "method_set",
        }
        and pd.api.types.is_numeric_dtype(metrics[c])
    ]
    agg = metrics.groupby(["semisynthetic", "split", "method"])[value_cols].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c).strip("_") for c in agg.columns.to_flat_index()]
    agg.reset_index().to_csv(out_dir / "metric_mean_std_by_split_method.csv", index=False)
    overall = metrics.groupby(["semisynthetic", "method"])[value_cols].agg(["mean", "std", "count"])
    overall.columns = ["_".join(c).strip("_") for c in overall.columns.to_flat_index()]
    overall.reset_index().to_csv(out_dir / "metric_mean_std_overall_by_method.csv", index=False)
    # Ranking distortion: compare exact-MAE ranking against censor-aware NLL ranking.
    rows = []
    for (semi, split, seed), g in metrics.dropna(subset=["mae_exact", "censored_nll"]).groupby(["semisynthetic", "split", "seed"]):
        if g["method"].nunique() < 3:
            continue
        mae_rank = g.set_index("method")["mae_exact"].rank(ascending=True)
        nll_rank = g.set_index("method")["censored_nll"].rank(ascending=True)
        common = mae_rank.index.intersection(nll_rank.index)
        if len(common) < 3:
            continue
        kt = kendalltau(mae_rank.loc[common], nll_rank.loc[common]).correlation
        sp = spearmanr(mae_rank.loc[common], nll_rank.loc[common]).correlation
        rows.append(
            {
                "semisynthetic": semi,
                "split": split,
                "seed": seed,
                "kendall_rank_corr_mae_vs_nll": float(kt),
                "spearman_rank_corr_mae_vs_nll": float(sp),
                "top1_mae": str(mae_rank.idxmin()),
                "top1_nll": str(nll_rank.idxmin()),
                "top1_changed": bool(mae_rank.idxmin() != nll_rank.idxmin()),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "ranking_distortion.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--artifacts-root", default="artifacts/benchmark")
    parser.add_argument("--splits", nargs="+", default=SPLITS_DEFAULT)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    parser.add_argument("--workers", type=int, default=max(1, min(64, (os.cpu_count() or 4) - 30)))
    parser.add_argument("--n-bits", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only-audit", action="store_true")
    parser.add_argument("--semisynthetic", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--method-set", choices=["all", "core", "trees", "ridge", "calibrated"], default="all")
    parser.add_argument("--tree-preset", choices=["standard", "fast"], default="standard")
    parser.add_argument("--permute-metadata-seed", type=int, default=None)
    args = parser.parse_args()

    root = ensure_dir(args.artifacts_root)
    df = pd.read_parquet(args.data)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=20260811).reset_index(drop=True)
    df = df.reset_index(drop=True)
    audit_tables(df, root / "audits")
    split_roles = {(split, seed): add_split(df, split, seed) for split in args.splits for seed in args.seeds}
    split_leakage_audit(df, split_roles, root / "audits")
    if args.only_audit:
        print(f"Wrote audits to {root / 'audits'}")
        return

    feature_store = build_features(df, root / "cache", args.n_bits, args.workers, permute_metadata_seed=args.permute_metadata_seed)
    all_metrics = []
    for seed in args.seeds:
        for split in args.splits:
            print(f"[real] split={split} seed={seed}")
            all_metrics.append(run_one_case(df, feature_store, split, seed, root, args.workers, args.device, args.quick, semisynthetic=False, method_set=args.method_set, tree_preset=args.tree_preset))
    if args.semisynthetic:
        semi = make_semisynthetic(df, seed=20260811)
        audit_tables(semi, root / "audits_semisynthetic")
        semi_features = build_features(semi, root / "cache_semi", args.n_bits, args.workers, permute_metadata_seed=args.permute_metadata_seed)
        semi_roles = {(split, seed): add_split(semi, split, seed) for split in args.splits for seed in args.seeds}
        split_leakage_audit(semi, semi_roles, root / "audits_semisynthetic")
        for seed in args.seeds:
            for split in args.splits:
                print(f"[semi] split={split} seed={seed}")
                all_metrics.append(run_one_case(semi, semi_features, split, seed, root, args.workers, args.device, args.quick, semisynthetic=True, method_set=args.method_set, tree_preset=args.tree_preset))
    metrics = pd.concat([m for m in all_metrics if m is not None and not m.empty], ignore_index=True) if all_metrics else pd.DataFrame()
    summarize_metrics(metrics, root)
    write_json(
        root / "suite_manifest.json",
        {
            "data": str(args.data),
            "artifacts_root": str(root),
            "rows": int(len(df)),
            "splits": args.splits,
            "seeds": args.seeds,
            "workers": int(args.workers),
            "n_bits": int(args.n_bits),
            "target_identity_encoding": "collision-free one-hot over sorted target-endpoint task IDs",
            "device": args.device,
            "quick": bool(args.quick),
            "semisynthetic": bool(args.semisynthetic),
            "method_set": args.method_set,
            "tree_preset": args.tree_preset,
            "permute_metadata_seed": args.permute_metadata_seed,
        },
    )
    print(f"Wrote suite to {root}")


if __name__ == "__main__":
    main()
