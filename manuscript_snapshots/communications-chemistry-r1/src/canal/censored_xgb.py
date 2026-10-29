from __future__ import annotations

import math

import numpy as np
import xgboost as xgb
from scipy import sparse
from scipy.special import ndtr


SQRT_2PI = math.sqrt(2.0 * math.pi)


class CensoredXGBRegressor:
    def __init__(self, booster: xgb.Booster, sigma: float):
        self.booster = booster
        self.sigma = sigma

    def predict(self, x):
        dmat = xgb.DMatrix(x)
        return self.booster.predict(dmat)


def _pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / SQRT_2PI


def censored_grad_hess(
    pred: np.ndarray,
    y: np.ndarray,
    relation: np.ndarray,
    sigma: float,
    censor_weight: float = 1.0,
    censored_anchor_weight: float = 0.0,
    right_censor_weight: float | None = None,
    right_censored_anchor_weight: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sigma = max(float(sigma), 1e-3)
    left_censor_weight = float(censor_weight)
    right_censor_weight = left_censor_weight if right_censor_weight is None else float(right_censor_weight)
    left_anchor_weight = float(censored_anchor_weight)
    right_anchor_weight = (
        left_anchor_weight if right_censored_anchor_weight is None else float(right_censored_anchor_weight)
    )
    y = y.astype(float)
    rel = relation.astype(str)
    z = np.clip((y - pred) / sigma, -12.0, 12.0)
    pdf = _pdf(z)

    grad = (pred - y) / (sigma * sigma)
    hess = np.full_like(grad, 1.0 / (sigma * sigma), dtype=float)

    left = rel == "<"
    if np.any(left):
        cdf = np.clip(ndtr(z[left]), 1e-12, 1.0)
        lam = pdf[left] / cdf
        grad[left] = left_censor_weight * lam / sigma
        hess[left] = np.maximum(left_censor_weight * lam * (z[left] + lam) / (sigma * sigma), 1e-6)

    right = rel == ">"
    if np.any(right):
        surv = np.clip(ndtr(-z[right]), 1e-12, 1.0)
        lam = pdf[right] / surv
        grad[right] = -right_censor_weight * lam / sigma
        hess[right] = np.maximum(right_censor_weight * lam * (lam - z[right]) / (sigma * sigma), 1e-6)

    if left_anchor_weight > 0.0 and np.any(left):
        anchor_grad = (pred[left] - y[left]) / (sigma * sigma)
        anchor_hess = np.full_like(anchor_grad, 1.0 / (sigma * sigma), dtype=float)
        grad[left] += left_anchor_weight * anchor_grad
        hess[left] += left_anchor_weight * anchor_hess

    if right_anchor_weight > 0.0 and np.any(right):
        anchor_grad = (pred[right] - y[right]) / (sigma * sigma)
        anchor_hess = np.full_like(anchor_grad, 1.0 / (sigma * sigma), dtype=float)
        grad[right] += right_anchor_weight * anchor_grad
        hess[right] += right_anchor_weight * anchor_hess

    return grad.astype(np.float32), hess.astype(np.float32)


def train_censored_xgb(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    relation_train: np.ndarray,
    seed: int,
    n_jobs: int,
    n_estimators: int,
    sigma: float | None = None,
    censor_weight: float = 1.0,
    censored_anchor_weight: float = 0.0,
    right_censor_weight: float | None = None,
    right_censored_anchor_weight: float | None = None,
) -> CensoredXGBRegressor:
    exact = relation_train.astype(str) == "="
    if sigma is None:
        base_values = y_train[exact] if np.any(exact) else y_train
        sigma = float(np.std(base_values))
    sigma = max(float(sigma), 0.1)
    base_score = float(np.median(y_train[exact] if np.any(exact) else y_train))

    dtrain = xgb.DMatrix(x_train, label=y_train)
    rel = relation_train.astype(str).copy()

    def objective(preds: np.ndarray, dmat: xgb.DMatrix):
        labels = dmat.get_label()
        return censored_grad_hess(
            preds,
            labels,
            rel,
            sigma,
            censor_weight=censor_weight,
            censored_anchor_weight=censored_anchor_weight,
            right_censor_weight=right_censor_weight,
            right_censored_anchor_weight=right_censored_anchor_weight,
        )

    params = {
        "max_depth": 6,
        "eta": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "tree_method": "hist",
        "lambda": 1.0,
        "seed": seed,
        "nthread": n_jobs,
        "base_score": base_score,
        "disable_default_eval_metric": 1,
    }
    booster = xgb.train(params, dtrain, num_boost_round=n_estimators, obj=objective, verbose_eval=False)
    return CensoredXGBRegressor(booster, sigma=sigma)
