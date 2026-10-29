from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor

from canal.censored_xgb import censored_grad_hess


class CenteredLGBMRegressor:
    def __init__(self, model: LGBMRegressor, base_score: float, sigma: float):
        self.model = model
        self.base_score = float(base_score)
        self.sigma = float(sigma)

    def predict(self, x):
        return self.base_score + self.model.predict(x)


def train_censored_lgbm(
    x_train,
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
) -> CenteredLGBMRegressor:
    rel = relation_train.astype(str).copy()
    exact = rel == "="
    base_values = y_train[exact] if np.any(exact) else y_train
    if sigma is None:
        sigma = float(np.std(base_values))
    sigma = max(float(sigma), 0.1)
    base_score = float(np.median(base_values))
    y_centered = y_train.astype(float) - base_score

    def objective(labels: np.ndarray, preds: np.ndarray):
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

    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=63,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective=objective,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
    )
    model.fit(x_train, y_centered)
    return CenteredLGBMRegressor(model=model, base_score=base_score, sigma=sigma)
