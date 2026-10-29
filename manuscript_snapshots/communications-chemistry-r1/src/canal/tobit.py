from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import ndtr


SQRT_2PI_LOG = 0.5 * math.log(2.0 * math.pi)


def _pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


class LinearTobitRegressor:
    """Gaussian censored linear regression with optional ridge penalty.

    The model predicts `mu = intercept + X @ coef` and maximizes the exact
    Gaussian likelihood plus left/right censored likelihood terms. The sigma is
    fixed from exact-label training residual scale to keep the baseline stable
    and comparable with the rest of the CANAL protocol.
    """

    def __init__(self, coef: np.ndarray, intercept: float, sigma: float):
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.sigma = float(sigma)

    def predict(self, x) -> np.ndarray:
        return np.asarray(x @ self.coef, dtype=np.float64).ravel() + self.intercept


def _as_dense_array(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray().astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)


def _standardize_train_val_test(x_train, x_val=None, x_test=None):
    train = _as_dense_array(x_train)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train = (train - mean) / scale
    outputs = [train]
    for x in [x_val, x_test]:
        if x is None:
            outputs.append(None)
        else:
            outputs.append((_as_dense_array(x) - mean) / scale)
    return tuple(outputs)


def standardize_train_val_test(x_train, x_val=None, x_test=None):
    return _standardize_train_val_test(x_train, x_val, x_test)


def _loss_grad(
    params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    rel: np.ndarray,
    sigma: float,
    ridge_alpha: float,
) -> tuple[float, np.ndarray]:
    coef = params[:-1]
    intercept = params[-1]
    mu = x @ coef + intercept
    sigma = max(float(sigma), 1e-4)

    loss_terms = np.zeros_like(y, dtype=np.float64)
    grad_mu = np.zeros_like(y, dtype=np.float64)

    exact = rel == "="
    if np.any(exact):
        err = mu[exact] - y[exact]
        loss_terms[exact] = SQRT_2PI_LOG + math.log(sigma) + 0.5 * (err / sigma) ** 2
        grad_mu[exact] = err / (sigma * sigma)

    left = rel == "<"
    if np.any(left):
        z = np.clip((y[left] - mu[left]) / sigma, -12.0, 12.0)
        cdf = np.clip(ndtr(z), 1e-12, 1.0)
        lam = _pdf(z) / cdf
        loss_terms[left] = -np.log(cdf)
        grad_mu[left] = lam / sigma

    right = rel == ">"
    if np.any(right):
        z = np.clip((y[right] - mu[right]) / sigma, -12.0, 12.0)
        surv = np.clip(ndtr(-z), 1e-12, 1.0)
        lam = _pdf(z) / surv
        loss_terms[right] = -np.log(surv)
        grad_mu[right] = -lam / sigma

    n = max(1, len(y))
    loss = float(loss_terms.mean() + 0.5 * ridge_alpha * np.dot(coef, coef) / n)
    grad_coef = x.T @ grad_mu / n + ridge_alpha * coef / n
    grad_intercept = np.array([grad_mu.mean()], dtype=np.float64)
    grad = np.concatenate([grad_coef, grad_intercept])
    return loss, grad


def train_linear_tobit(
    x_train,
    y_train: np.ndarray,
    relation_train: np.ndarray,
    sigma: float | None = None,
    ridge_alpha: float = 1.0,
    max_iter: int = 300,
) -> LinearTobitRegressor:
    x = _as_dense_array(x_train)
    y = np.asarray(y_train, dtype=np.float64)
    rel = np.asarray(relation_train).astype(str)
    exact = rel == "="
    if not np.any(exact):
        exact = np.ones_like(rel, dtype=bool)
    if sigma is None:
        sigma = float(np.std(y[exact]))
    sigma = max(float(sigma), 0.1)

    x_exact = x[exact]
    y_exact = y[exact]
    x_aug = np.hstack([x_exact, np.ones((len(x_exact), 1), dtype=np.float64)])
    penalty = np.eye(x_aug.shape[1], dtype=np.float64) * float(ridge_alpha)
    penalty[-1, -1] = 0.0
    try:
        init = np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y_exact)
    except np.linalg.LinAlgError:
        init = np.linalg.lstsq(x_aug.T @ x_aug + penalty, x_aug.T @ y_exact, rcond=None)[0]

    def objective(params: np.ndarray):
        return _loss_grad(params, x, y, rel, sigma, float(ridge_alpha))

    opt = minimize(
        objective,
        init,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iter), "ftol": 1e-8, "gtol": 1e-5, "maxls": 30},
    )
    params = opt.x if opt.success or np.isfinite(opt.fun) else init
    return LinearTobitRegressor(coef=params[:-1], intercept=float(params[-1]), sigma=sigma)
