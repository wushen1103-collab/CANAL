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

from run_benchmark import (
    SEEDS_DEFAULT,
    SPLITS_DEFAULT,
    add_split,
    censor_arrays,
    ensure_dir,
    eval_predictions,
    fit_censored_affine_calibration,
    fit_sigma,
    train_sklearn_model,
    build_features,
)


PRIMARY = "canal_extratrees_molecule_calibrated"
BASELINE = "extratrees_molecule_only_naive"


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
        # For the aggregate row, average split-specific observations within the
        # same inferential unit instead of treating target--split cells as
        # independent. Split-specific rows remove only the fixed split label.
        index_cols = [c for c in unit_cols if c != "split"]
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


def boundary_probability(mu: np.ndarray, sigma: float, y: np.ndarray, ctype: np.ndarray) -> np.ndarray:
    z = (y - mu) / (sigma + 1e-6)
    # upper: latent <= threshold; lower: latent >= threshold
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


def exact_interval_coverage(mu: np.ndarray, sigma: float, y: np.ndarray, ctype: np.ndarray) -> tuple[float, float]:
    exact = ctype == 0
    if exact.sum() == 0:
        return np.nan, np.nan
    err = np.abs(y[exact] - mu[exact])
    return float((err <= 0.67449 * sigma).mean()), float((err <= 1.64485 * sigma).mean())


def reliability_rows(split: str, seed: int, method: str, mu: np.ndarray, sigma: float, y: np.ndarray, ctype: np.ndarray) -> list[dict]:
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
                    "method": method,
                    "bin_low": float(bins[b]),
                    "bin_high": float(bins[b + 1]),
                    "n_censored": int(m.sum()),
                    "predicted_boundary_probability": float(p[mask][m].mean()),
                    "point_boundary_satisfaction": float(sat[mask][m].mean()),
                    "abs_gap": float(abs(p[mask][m].mean() - sat[mask][m].mean())),
                }
            )
    cov50, cov90 = exact_interval_coverage(mu, sigma, y, ctype)
    rows.append(
        {
            "split": split,
            "seed": seed,
            "method": method,
            "bin_low": "exact_coverage",
            "bin_high": "exact_coverage",
            "n_censored": int((ctype == 0).sum()),
            "predicted_boundary_probability": 0.50,
            "point_boundary_satisfaction": cov50,
            "abs_gap": abs(cov50 - 0.50) if np.isfinite(cov50) else np.nan,
            "coverage_level": 0.50,
        }
    )
    rows.append(
        {
            "split": split,
            "seed": seed,
            "method": method,
            "bin_low": "exact_coverage",
            "bin_high": "exact_coverage",
            "n_censored": int((ctype == 0).sum()),
            "predicted_boundary_probability": 0.90,
            "point_boundary_satisfaction": cov90,
            "abs_gap": abs(cov90 - 0.90) if np.isfinite(cov90) else np.nan,
            "coverage_level": 0.90,
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/chembl37_canal20.parquet")
    parser.add_argument("--out", default="artifacts/paired_calibration")
    parser.add_argument("--splits", nargs="+", default=SPLITS_DEFAULT)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--tree-preset", choices=["standard", "fast"], default="fast")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    if args.summarize_only:
        target_df = pd.read_csv(out_dir / "target_level_metrics.csv")
        summarize_paired(
            target_df,
            ["target_endpoint", "split", "seed"],
            out_dir / "paired_stats_target_split_seed.csv",
            "target_endpoint_split_seed",
        )
        seed_agg = target_df.groupby(["target_endpoint", "split", "method"], as_index=False)[
            ["mae_exact", "censored_nll", "spearman_exact"]
        ].mean()
        summary = summarize_paired(
            seed_agg,
            ["target_endpoint", "split"],
            out_dir / "paired_stats_target_seed_aggregated.csv",
            "target_endpoint_after_seed_average",
        )
        print(summary.to_json(orient="records", indent=2))
        return

    df = pd.read_parquet(args.data)
    features = build_features(df, out_dir / "cache", args.n_bits, args.workers)
    x = features.matrix("molecule_only")
    y, ctype = censor_arrays(df)
    target_values = sorted(df["target_endpoint"].astype(str).unique())
    target_arr = df["target_endpoint"].astype(str).to_numpy()
    target_rows = []
    calibration_rows = []
    run_rows = []
    for seed in args.seeds:
        for split in args.splits:
            print(f"[paired-target] split={split} seed={seed}", flush=True)
            roles = add_split(df, split, seed)
            idx_train = np.flatnonzero(roles.eq("train").to_numpy())
            idx_val = np.flatnonzero(roles.eq("val").to_numpy())
            idx_test = np.flatnonzero(roles.eq("test").to_numpy())
            model = train_sklearn_model("extratrees", x[idx_train], y[idx_train], workers=args.workers, seed=seed, tree_preset=args.tree_preset)
            raw_val = model.predict(x[idx_val]).astype(float)
            raw_test = model.predict(x[idx_test]).astype(float)
            sigma_raw = fit_sigma(y[idx_val], raw_val, ctype[idx_val])
            scale, bias, sigma_cal, val_nll_cal = fit_censored_affine_calibration(y[idx_val], raw_val, ctype[idx_val])
            cal_test = scale * raw_test + bias
            run_rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "calibration_scope": "split_seed_global_validation_only",
                    "train_n": int(len(idx_train)),
                    "val_n": int(len(idx_val)),
                    "test_n": int(len(idx_test)),
                    "calibration_scale": scale,
                    "calibration_bias": bias,
                    "sigma_raw": sigma_raw,
                    "sigma_calibrated": sigma_cal,
                    "validation_censored_nll_calibrated": val_nll_cal,
                }
            )
            test_targets = target_arr[idx_test]
            for target in target_values:
                mask = test_targets == target
                if mask.sum() < 10:
                    continue
                local_idx = np.flatnonzero(mask)
                for method, pred, sigma in [(BASELINE, raw_test, sigma_raw), (PRIMARY, cal_test, sigma_cal)]:
                    metrics = eval_predictions(pred[local_idx], y[idx_test][local_idx], ctype[idx_test][local_idx], sigma=sigma)
                    target_rows.append(
                        {
                            "target_endpoint": target,
                            "split": split,
                            "seed": seed,
                            "method": method,
                            "n_test_target": int(mask.sum()),
                            **metrics,
                        }
                    )
            calibration_rows.extend(reliability_rows(split, seed, PRIMARY, cal_test, sigma_cal, y[idx_test], ctype[idx_test]))

    target_df = pd.DataFrame(target_rows)
    target_df.to_csv(out_dir / "target_level_metrics.csv", index=False)
    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(out_dir / "calibration_fit_audit.csv", index=False)
    cal_df = pd.DataFrame(calibration_rows)
    cal_df.to_csv(out_dir / "calibration_reliability_bins.csv", index=False)
    target_summary = summarize_paired(target_df, ["target_endpoint", "split", "seed"], out_dir / "paired_stats_target_split_seed.csv", "target_endpoint_split_seed")
    seed_agg = target_df.groupby(["target_endpoint", "split", "method"], as_index=False)[["mae_exact", "censored_nll", "spearman_exact"]].mean()
    seed_agg_summary = summarize_paired(seed_agg, ["target_endpoint", "split"], out_dir / "paired_stats_target_seed_aggregated.csv", "target_endpoint_after_seed_average")

    cal_only = cal_df[cal_df["bin_low"].ne("exact_coverage")].copy()
    exact_cov = cal_df[cal_df["bin_low"].eq("exact_coverage")].copy()
    diagnostic = {
        "calibration_fit_protocol": "For each split and seed, ExtraTrees is fitted on train rows only; affine scale/bias/sigma are fitted on validation rows only; test rows are used only after calibration is fixed.",
        "calibration_scope": "split_seed_global_validation_only",
        "target_level_metric_rows": int(len(target_df)),
        "paired_target_split_seed_rows": int(len(target_summary)),
        "paired_target_seed_aggregated_rows": int(len(seed_agg_summary)),
        "boundary_probability_ece_like_abs_gap": float(np.average(cal_only["abs_gap"], weights=cal_only["n_censored"])) if len(cal_only) else None,
        "exact_coverage_mean": exact_cov.groupby("coverage_level")["point_boundary_satisfaction"].mean().to_dict() if len(exact_cov) else {},
        "run_fit_audit_csv": str(out_dir / "calibration_fit_audit.csv"),
    }
    write_json(out_dir / "paired_target_calibration_summary.json", diagnostic)
    print(json.dumps(diagnostic, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
