#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtr
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from run_benchmark import (
    SEEDS_DEFAULT,
    SPLITS_DEFAULT,
    add_split,
    build_features,
    censor_arrays,
    ensure_dir,
    eval_predictions,
    fit_censored_affine_calibration,
    fit_sigma,
    gaussian_censored_nll,
    macro_spearman,
    train_sklearn_model,
)


BASELINE = "extratrees_molecule_only_naive"
PRIMARY = "canal_extratrees_molecule_targetwise_calibrated"


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ci_mean(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    se = x.std(ddof=1) / math.sqrt(len(x))
    tcrit = stats.t.ppf(0.975, len(x) - 1)
    return float(x.mean() - tcrit * se), float(x.mean() + tcrit * se)


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    out = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        out[idx] = min(1.0, running)
    return out.tolist()


def summarize_paired(df: pd.DataFrame, unit_cols: list[str], out_path: Path, unit_name: str) -> pd.DataFrame:
    rows = []
    metrics = ["mae_exact", "censored_nll", "spearman_exact"]
    for split in ["overall"] + sorted(df["split"].unique()):
        sub = df if split == "overall" else df[df["split"].eq(split)]
        index_cols = unit_cols if split == "overall" else [c for c in unit_cols if c != "split"]
        wide = sub.pivot_table(index=index_cols, columns="method", values=metrics)
        for metric in metrics:
            if (metric, PRIMARY) not in wide.columns or (metric, BASELINE) not in wide.columns:
                continue
            delta = (wide[(metric, PRIMARY)] - wide[(metric, BASELINE)]).dropna().to_numpy(dtype=float)
            if len(delta) == 0:
                continue
            lo, hi = ci_mean(delta)
            try:
                p_w = float(stats.wilcoxon(delta).pvalue) if len(delta) >= 2 and not np.allclose(delta, 0) else 1.0
            except ValueError:
                p_w = 1.0
            p_t = float(stats.ttest_1samp(delta, 0.0).pvalue) if len(delta) >= 2 else np.nan
            improve = delta < 0 if metric != "spearman_exact" else delta > 0
            rows.append(
                {
                    "paired_unit": unit_name,
                    "split": split,
                    "metric": metric,
                    "n_pairs": int(len(delta)),
                    "baseline_mean": float(wide[(metric, BASELINE)].mean()),
                    "primary_mean": float(wide[(metric, PRIMARY)].mean()),
                    "delta_mean": float(delta.mean()),
                    "delta_std": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                    "delta_ci95_low": lo,
                    "delta_ci95_high": hi,
                    "win_rate": float(improve.mean()),
                    "wilcoxon_p": p_w,
                    "ttest_p": p_t,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["wilcoxon_p_holm"] = holm(out["wilcoxon_p"].fillna(1.0).tolist())
    out.to_csv(out_path, index=False)
    return out


def boundary_probability(mu: np.ndarray, sigma: np.ndarray | float, y: np.ndarray, ctype: np.ndarray) -> np.ndarray:
    sigma_arr = np.asarray(sigma, dtype=float) + 1e-6
    z = (y - mu) / sigma_arr
    p = np.full(len(mu), np.nan, dtype=float)
    upper = ctype == 1
    lower = ctype == 2
    p[upper] = ndtr(z[upper])
    p[lower] = 1.0 - ndtr(z[lower])
    return p


def point_satisfies(mu: np.ndarray, y: np.ndarray, ctype: np.ndarray) -> np.ndarray:
    out = np.full(len(mu), np.nan, dtype=float)
    upper = ctype == 1
    lower = ctype == 2
    out[upper] = (mu[upper] <= y[upper]).astype(float)
    out[lower] = (mu[lower] >= y[lower]).astype(float)
    return out


def eval_with_sigma_array(mu: np.ndarray, y: np.ndarray, ctype: np.ndarray, sigma: np.ndarray, groups: np.ndarray | None = None) -> dict:
    exact = ctype == 0
    out: dict[str, float] = {"sigma": float(np.mean(sigma))}
    if exact.any():
        out["mae_exact"] = float(mean_absolute_error(y[exact], mu[exact]))
        out["rmse_exact"] = float(math.sqrt(mean_squared_error(y[exact], mu[exact])))
        out["spearman_exact_pooled"] = float(spearmanr(y[exact], mu[exact]).correlation) if exact.sum() > 3 else np.nan
        if groups is not None:
            out["spearman_exact"], out["spearman_exact_task_count"] = macro_spearman(
                y[exact], mu[exact], np.asarray(groups)[exact]
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
    out["censored_nll"] = gaussian_censored_nll(mu, sigma, y, ctype)
    cens = ctype != 0
    if cens.any():
        sat = point_satisfies(mu, y, ctype)
        prob = boundary_probability(mu, sigma, y, ctype)
        out["boundary_satisfaction"] = float(np.nanmean(sat[cens]))
        out["boundary_prob_mean"] = float(np.nanmean(prob[cens]))
        out["boundary_prob_ece_proxy"] = float(abs(np.nanmean(prob[cens]) - np.nanmean(sat[cens])))
    else:
        out["boundary_satisfaction"] = np.nan
        out["boundary_prob_mean"] = np.nan
        out["boundary_prob_ece_proxy"] = np.nan
    return out


def reliability_rows(split: str, seed: int, target: str, mu: np.ndarray, sigma: float, y: np.ndarray, ctype: np.ndarray) -> list[dict]:
    p = boundary_probability(mu, sigma, y, ctype)
    sat = point_satisfies(mu, y, ctype)
    mask = np.isfinite(p) & np.isfinite(sat)
    rows = []
    if mask.sum() > 0:
        bins = np.linspace(0.0, 1.0, 11)
        ids = np.clip(np.digitize(p[mask], bins, right=True) - 1, 0, 9)
        for b in range(10):
            m = ids == b
            if not m.any():
                continue
            rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "target_endpoint": target,
                    "method": PRIMARY,
                    "bin_low": float(bins[b]),
                    "bin_high": float(bins[b + 1]),
                    "n_censored": int(m.sum()),
                    "predicted_boundary_probability": float(p[mask][m].mean()),
                    "point_boundary_satisfaction": float(sat[mask][m].mean()),
                    "abs_gap": float(abs(p[mask][m].mean() - sat[mask][m].mean())),
                }
            )
    exact = ctype == 0
    if exact.any():
        err = np.abs(y[exact] - mu[exact])
        cov50 = float((err <= 0.67449 * sigma).mean())
        cov90 = float((err <= 1.64485 * sigma).mean())
        rows.extend(
            [
                {
                    "split": split,
                    "seed": seed,
                    "target_endpoint": target,
                    "method": PRIMARY,
                    "bin_low": "exact_coverage",
                    "bin_high": "exact_coverage",
                    "n_censored": int(exact.sum()),
                    "predicted_boundary_probability": 0.50,
                    "point_boundary_satisfaction": cov50,
                    "abs_gap": abs(cov50 - 0.50),
                    "coverage_level": 0.50,
                },
                {
                    "split": split,
                    "seed": seed,
                    "target_endpoint": target,
                    "method": PRIMARY,
                    "bin_low": "exact_coverage",
                    "bin_high": "exact_coverage",
                    "n_censored": int(exact.sum()),
                    "predicted_boundary_probability": 0.90,
                    "point_boundary_satisfaction": cov90,
                    "abs_gap": abs(cov90 - 0.90),
                    "coverage_level": 0.90,
                },
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/chembl37_canal20.parquet")
    parser.add_argument("--out", default="artifacts/targetwise_calibration")
    parser.add_argument("--splits", nargs="+", default=SPLITS_DEFAULT)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--tree-preset", choices=["standard", "fast"], default="fast")
    parser.add_argument("--min-val-target-rows", type=int, default=30)
    parser.add_argument("--min-test-target-rows", type=int, default=10)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    df = pd.read_parquet(args.data)
    features = build_features(df, out_dir / "cache", args.n_bits, args.workers)
    x = features.matrix("molecule_only")
    y, ctype = censor_arrays(df)
    target_values = sorted(df["target_endpoint"].astype(str).unique())
    target_arr = df["target_endpoint"].astype(str).to_numpy()

    target_rows = []
    calibration_rows = []
    reliability = []
    case_rows = []

    for seed in args.seeds:
        for split in args.splits:
            print(f"[targetwise-calibration] split={split} seed={seed}", flush=True)
            roles = add_split(df, split, seed)
            idx_train = np.flatnonzero(roles.eq("train").to_numpy())
            idx_val = np.flatnonzero(roles.eq("val").to_numpy())
            idx_test = np.flatnonzero(roles.eq("test").to_numpy())

            model = train_sklearn_model("extratrees", x[idx_train], y[idx_train], workers=args.workers, seed=seed, tree_preset=args.tree_preset)
            raw_val = model.predict(x[idx_val]).astype(float)
            raw_test = model.predict(x[idx_test]).astype(float)
            sigma_raw = fit_sigma(y[idx_val], raw_val, ctype[idx_val])

            targetwise_test = raw_test.copy()
            targetwise_sigma = np.full(len(idx_test), sigma_raw, dtype=float)
            val_targets = target_arr[idx_val]
            test_targets = target_arr[idx_test]

            for target in target_values:
                val_mask = val_targets == target
                test_mask = test_targets == target
                val_n = int(val_mask.sum())
                test_n = int(test_mask.sum())
                if val_n >= args.min_val_target_rows:
                    scale, bias, sigma_cal, val_nll_cal = fit_censored_affine_calibration(y[idx_val][val_mask], raw_val[val_mask], ctype[idx_val][val_mask])
                    fallback = False
                else:
                    scale, bias, sigma_cal = 1.0, 0.0, sigma_raw
                    val_nll_cal = gaussian_censored_nll(raw_val[val_mask], sigma_raw, y[idx_val][val_mask], ctype[idx_val][val_mask]) if val_n else np.nan
                    fallback = True
                if test_n > 0:
                    targetwise_test[test_mask] = scale * raw_test[test_mask] + bias
                    targetwise_sigma[test_mask] = sigma_cal
                calibration_rows.append(
                    {
                        "split": split,
                        "seed": seed,
                        "target_endpoint": target,
                        "calibration_scope": "target_endpoint_split_seed_validation_only",
                        "train_n": int(len(idx_train)),
                        "val_target_n": val_n,
                        "test_target_n": test_n,
                        "scale": scale,
                        "bias": bias,
                        "sigma_calibrated": sigma_cal,
                        "validation_censored_nll_calibrated": val_nll_cal,
                        "fallback_identity": fallback,
                    }
                )

                if test_n >= args.min_test_target_rows:
                    local_idx = np.flatnonzero(test_mask)
                    metrics_base = eval_predictions(raw_test[local_idx], y[idx_test][local_idx], ctype[idx_test][local_idx], sigma=sigma_raw)
                    metrics_cal = eval_predictions(
                        targetwise_test[local_idx],
                        y[idx_test][local_idx],
                        ctype[idx_test][local_idx],
                        sigma=sigma_cal,
                    )
                    target_rows.append({"target_endpoint": target, "split": split, "seed": seed, "method": BASELINE, "n_test_target": test_n, **metrics_base})
                    target_rows.append({"target_endpoint": target, "split": split, "seed": seed, "method": PRIMARY, "n_test_target": test_n, **metrics_cal})
                    reliability.extend(reliability_rows(split, seed, target, targetwise_test[local_idx], sigma_cal, y[idx_test][local_idx], ctype[idx_test][local_idx]))

            y_test = y[idx_test]
            c_test = ctype[idx_test]
            case_rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "method": BASELINE,
                    **eval_predictions(raw_test, y_test, c_test, sigma=sigma_raw, groups=test_targets),
                }
            )
            case_rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "method": PRIMARY,
                    **eval_with_sigma_array(targetwise_test, y_test, c_test, targetwise_sigma, groups=test_targets),
                }
            )

    target_df = pd.DataFrame(target_rows)
    target_df.to_csv(out_dir / "target_level_metrics.csv", index=False)
    case_df = pd.DataFrame(case_rows)
    case_df.to_csv(out_dir / "split_seed_metrics.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(out_dir / "targetwise_calibration_fit_audit.csv", index=False)
    cal_df = pd.DataFrame(reliability)
    cal_df.to_csv(out_dir / "targetwise_calibration_reliability_bins.csv", index=False)

    target_summary = summarize_paired(target_df, ["target_endpoint", "split", "seed"], out_dir / "paired_stats_target_split_seed.csv", "target_endpoint_split_seed")
    seed_agg = target_df.groupby(["target_endpoint", "split", "method"], as_index=False)[["mae_exact", "censored_nll", "spearman_exact"]].mean()
    seed_summary = summarize_paired(seed_agg, ["target_endpoint", "split"], out_dir / "paired_stats_target_seed_aggregated.csv", "target_endpoint_after_seed_average")
    split_summary = summarize_paired(case_df, ["split", "seed"], out_dir / "paired_stats_split_seed.csv", "split_seed_aggregate")

    cal_only = cal_df[cal_df["bin_low"].ne("exact_coverage")].copy()
    exact_cov = cal_df[cal_df["bin_low"].eq("exact_coverage")].copy()
    fit_df = pd.DataFrame(calibration_rows)
    diagnostic = {
        "calibration_fit_protocol": "For each split and seed, ExtraTrees is fitted on train rows only; affine scale/bias/sigma are fitted separately for each target_endpoint using validation rows only; test rows are used only after target-wise calibration is fixed.",
        "calibration_scope": "target_endpoint_split_seed_validation_only",
        "target_level_metric_rows": int(len(target_df)),
        "split_seed_metric_rows": int(len(case_df)),
        "targetwise_fit_rows": int(len(fit_df)),
        "fallback_identity_rows": int(fit_df["fallback_identity"].sum()) if len(fit_df) else 0,
        "paired_target_split_seed_rows": int(len(target_summary)),
        "paired_target_seed_aggregated_rows": int(len(seed_summary)),
        "paired_split_seed_rows": int(len(split_summary)),
        "boundary_probability_ece_like_abs_gap": float(np.average(cal_only["abs_gap"], weights=cal_only["n_censored"])) if len(cal_only) else None,
        "exact_coverage_mean": exact_cov.groupby("coverage_level")["point_boundary_satisfaction"].mean().to_dict() if len(exact_cov) else {},
        "targetwise_fit_audit_csv": str(out_dir / "targetwise_calibration_fit_audit.csv"),
    }
    write_json(out_dir / "targetwise_calibration_summary.json", diagnostic)
    print(json.dumps(diagnostic, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
