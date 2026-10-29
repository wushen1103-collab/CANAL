#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


BOOTSTRAP_SEED = 360036
BOOTSTRAP_REPLICATES = 20_000
CANAL = "censored_likelihood"
MATCHED = "censored_as_exact"


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def rank_correlation(group: pd.DataFrame) -> float:
    if len(group) < 5 or group["observed"].nunique() < 2 or group["predicted"].nunique() < 2:
        return float("nan")
    return float(spearmanr(group["observed"], group["predicted"]).statistic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="later_exact_s*.parquet")
    parser.add_argument("--chembl35", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.prediction_dir.glob(args.pattern))
    if len(files) != 5:
        raise RuntimeError(f"Expected five seed files, found {len(files)}: {files}")
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    expected = {"exact_only", MATCHED, CANAL}
    if set(frame["treatment"].unique()) != expected:
        raise RuntimeError(f"Unexpected treatments: {sorted(frame['treatment'].unique())}")
    if frame["seed"].nunique() != 5:
        raise RuntimeError("Expected five seeds")
    if not frame["p_relation"].eq("=").all():
        raise RuntimeError("The later-exact evaluation contains non-exact rows")

    pair = (
        frame.groupby(["target_chembl_id", "parent_inchikey", "seed", "treatment"])
        .agg(
            later_exact_records=("p_activity", "size"),
            observed=("p_activity", "median"),
            predicted=("pred", "median"),
            sigma=("sigma", "median"),
        )
        .reset_index()
    )
    pair["absolute_error"] = (pair["predicted"] - pair["observed"]).abs()
    pair["gaussian_nll"] = (
        0.5 * np.log(2 * np.pi * pair["sigma"] ** 2)
        + 0.5 * ((pair["observed"] - pair["predicted"]) / pair["sigma"]) ** 2
    )

    if args.chembl35:
        old = pd.read_parquet(args.chembl35)
        old = old.loc[old["split_s3"].eq("train") & old["p_relation"].ne("=")].copy()
        history = (
            old.groupby(["target_chembl_id", "parent_inchikey"])
            .agg(
                historical_censored_rows=("p_activity", "size"),
                historical_median_bound=("p_activity", "median"),
                historical_directions=("p_relation", lambda x: "".join(sorted(set(x)))),
            )
            .reset_index()
        )
        pair = pair.merge(
            history,
            on=["target_chembl_id", "parent_inchikey"],
            how="left",
            validate="many_to_one",
        )
        pair["followup_stratum"] = "mixed_direction"
        left = pair["historical_directions"].eq("<")
        right = pair["historical_directions"].eq(">")
        pair.loc[left & pair["observed"].lt(pair["historical_median_bound"]), "followup_stratum"] = "left_compatible"
        pair.loc[left & pair["observed"].ge(pair["historical_median_bound"]), "followup_stratum"] = "left_incompatible"
        pair.loc[right & pair["observed"].gt(pair["historical_median_bound"]), "followup_stratum"] = "right_compatible"
        pair.loc[right & pair["observed"].le(pair["historical_median_bound"]), "followup_stratum"] = "right_incompatible"
    else:
        pair["followup_stratum"] = "not_assessed"

    target_seed = (
        pair.groupby(["target_chembl_id", "seed", "treatment"])
        .agg(
            pairs=("parent_inchikey", "nunique"),
            mae=("absolute_error", "mean"),
            nll=("gaussian_nll", "mean"),
        )
        .reset_index()
    )
    ranks = (
        pair.groupby(["target_chembl_id", "seed", "treatment"])
        .apply(rank_correlation, include_groups=False)
        .rename("spearman")
        .reset_index()
    )
    target_seed = target_seed.merge(ranks, on=["target_chembl_id", "seed", "treatment"])
    summary = (
        target_seed.groupby("treatment")
        .agg(
            target_seed_cells=("mae", "size"),
            targets=("target_chembl_id", "nunique"),
            mean_target_balanced_mae=("mae", "mean"),
            sd_target_balanced_mae=("mae", "std"),
            mean_target_balanced_nll=("nll", "mean"),
            sd_target_balanced_nll=("nll", "std"),
            mean_spearman=("spearman", "mean"),
            sd_spearman=("spearman", "std"),
            rank_cells=("spearman", "count"),
        )
        .reset_index()
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    inference_rows = []
    for comparator in [MATCHED, "exact_only"]:
        for metric, lower_is_better in [("mae", True), ("nll", True), ("spearman", False)]:
            target = (
                target_seed.groupby(["target_chembl_id", "treatment"])[metric]
                .mean()
                .unstack()
                .dropna(subset=[comparator, CANAL])
            )
            delta = (target[CANAL] - target[comparator]).to_numpy(dtype=float)
            if len(delta):
                ci_low, ci_high = bootstrap_ci(delta, rng)
                p_value = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
                wins = int((delta < 0).sum() if lower_is_better else (delta > 0).sum())
            else:
                ci_low = ci_high = p_value = float("nan")
                wins = 0
            inference_rows.append(
                {
                    "comparator": comparator,
                    "metric": metric,
                    "experimental_unit": "target",
                    "n_targets": int(len(delta)),
                    "delta_canal_minus_comparator": float(delta.mean()) if len(delta) else float("nan"),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "wilcoxon_p": p_value,
                    "target_wins": wins,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair.to_csv(args.out_dir / "later_exact_pair_predictions.csv", index=False)
    target_seed.to_csv(args.out_dir / "later_exact_target_seed_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "later_exact_summary.csv", index=False)
    inference = pd.DataFrame(inference_rows)
    inference.to_csv(args.out_dir / "later_exact_target_inference.csv", index=False)

    stratum_target_seed = (
        pair.groupby(["followup_stratum", "target_chembl_id", "seed", "treatment"])
        .agg(pairs=("parent_inchikey", "nunique"), mae=("absolute_error", "mean"))
        .reset_index()
    )
    stratum_summary = (
        stratum_target_seed.groupby(["followup_stratum", "treatment"])
        .agg(
            target_seed_cells=("mae", "size"),
            targets=("target_chembl_id", "nunique"),
            mean_target_balanced_mae=("mae", "mean"),
            sd_target_balanced_mae=("mae", "std"),
        )
        .reset_index()
    )
    stratum_summary.to_csv(args.out_dir / "later_exact_stratum_summary.csv", index=False)
    stratum_inference_rows = []
    for stratum, block in stratum_target_seed.groupby("followup_stratum"):
        target = (
            block.groupby(["target_chembl_id", "treatment"])["mae"]
            .mean()
            .unstack()
            .dropna(subset=[MATCHED, CANAL])
        )
        delta = (target[CANAL] - target[MATCHED]).to_numpy(dtype=float)
        if len(delta):
            ci_low, ci_high = bootstrap_ci(delta, rng)
            p_value = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
        else:
            ci_low = ci_high = p_value = float("nan")
        stratum_inference_rows.append(
            {
                "followup_stratum": stratum,
                "n_targets": int(len(delta)),
                "delta_canal_minus_matched_mae": float(delta.mean()) if len(delta) else float("nan"),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "wilcoxon_p": p_value,
                "target_wins": int((delta < 0).sum()),
            }
        )
    stratum_inference = pd.DataFrame(stratum_inference_rows)
    stratum_inference.to_csv(args.out_dir / "later_exact_stratum_inference.csv", index=False)
    manifest = {
        "prediction_files": [str(path) for path in files],
        "rows": int(len(frame)),
        "pairs": int(pair[["target_chembl_id", "parent_inchikey"]].drop_duplicates().shape[0]),
        "targets": int(pair["target_chembl_id"].nunique()),
        "seeds": sorted(int(seed) for seed in pair["seed"].unique()),
        "pair_aggregation": "median observed and predicted pActivity across later exact records",
        "experimental_unit": "target",
        "followup_strata": (
            "Compatibility of the median later exact pActivity with the median historical "
            "ChEMBL35 S3-training bound; this is a descriptive post hoc analysis."
        ),
    }
    (args.out_dir / "later_exact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(inference.to_string(index=False))
    print(stratum_summary.to_string(index=False))
    print(stratum_inference.to_string(index=False))


if __name__ == "__main__":
    main()
