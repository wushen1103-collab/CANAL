from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr
from scipy.stats import spearmanr


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "spearman": np.nan}
    err = y_pred[mask] - y_true[mask]
    if mask.sum() >= 3:
        rho = spearmanr(y_true[mask], y_pred[mask]).statistic
    else:
        rho = np.nan
    return {
        "n": int(mask.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
    }


def gaussian_nll(y_true: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = max(float(sigma), 1e-4)
    mask = np.isfinite(y_true) & np.isfinite(mu)
    if mask.sum() == 0:
        return np.nan
    err = y_true[mask] - mu[mask]
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * (err / sigma) ** 2))


def interval_coverage(y_true: np.ndarray, mu: np.ndarray, sigma: float, level: float = 0.90) -> float:
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    z = 1.6448536269514722 if abs(level - 0.90) < 1e-9 else 1.96
    sigma = max(float(sigma), 1e-4)
    mask = np.isfinite(y_true) & np.isfinite(mu)
    if mask.sum() == 0:
        return np.nan
    lower = mu[mask] - z * sigma
    upper = mu[mask] + z * sigma
    return float(np.mean((y_true[mask] >= lower) & (y_true[mask] <= upper)))


def censored_gaussian_nll(threshold: np.ndarray, relation: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    """Average NLL for pActivity censoring.

    relation is in pActivity space:
    - '>' means latent activity exceeds the threshold.
    - '<' means latent activity is below the threshold.
    """

    threshold = np.asarray(threshold, dtype=float)
    relation = np.asarray(relation).astype(str)
    mu = np.asarray(mu, dtype=float)
    sigma = max(float(sigma), 1e-4)
    mask = np.isfinite(threshold) & np.isfinite(mu) & np.isin(relation, [">", "<"])
    if mask.sum() == 0:
        return np.nan
    z = (threshold[mask] - mu[mask]) / sigma
    cdf = np.clip(ndtr(z), 1e-12, 1 - 1e-12)
    probs = np.where(relation[mask] == ">", 1.0 - cdf, cdf)
    return float(np.mean(-np.log(np.clip(probs, 1e-12, 1.0))))


def censored_direction_accuracy(threshold: np.ndarray, relation: np.ndarray, mu: np.ndarray) -> float:
    threshold = np.asarray(threshold, dtype=float)
    relation = np.asarray(relation).astype(str)
    mu = np.asarray(mu, dtype=float)
    mask = np.isfinite(threshold) & np.isfinite(mu) & np.isin(relation, [">", "<"])
    if mask.sum() == 0:
        return np.nan
    ok = np.where(relation[mask] == ">", mu[mask] > threshold[mask], mu[mask] < threshold[mask])
    return float(np.mean(ok))
