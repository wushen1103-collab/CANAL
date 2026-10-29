#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


BOOTSTRAP_SEED = 350037
BOOTSTRAP_REPLICATES = 20_000
METRICS = {
    "test_mae": False,
    "test_censored_nll": False,
    "test_censored_dir_acc": True,
    "worst_assay_mae": False,
}


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_adjust(p_values: pd.Series) -> pd.Series:
    order = np.argsort(p_values.to_numpy(dtype=float))
    ranked = p_values.to_numpy(dtype=float)[order]
    adjusted_ranked = np.maximum.accumulate((len(ranked) - np.arange(len(ranked))) * ranked)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def read_many(paths: list[Path]) -> pd.DataFrame:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing result files: {missing}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def infer(
    frame: pd.DataFrame,
    group_columns: list[str],
    comparator: str,
    family: str,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    target = frame.groupby(group_columns + ["target", "treatment"])[list(METRICS)].mean().reset_index()
    rows: list[dict[str, object]] = []
    iterator = [((), target)] if not group_columns else target.groupby(group_columns, dropna=False)
    for keys, block in iterator:
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_values = dict(zip(group_columns, keys))
        for metric, higher_is_better in METRICS.items():
            wide = block.pivot(index="target", columns="treatment", values=metric).dropna(
                subset=[comparator, "canal_validation_ranked"]
            )
            delta = (wide["canal_validation_ranked"] - wide[comparator]).to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_ci(delta, rng)
            p_value = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
            wins = int((delta > 0).sum() if higher_is_better else (delta < 0).sum())
            rows.append(
                {
                    **group_values,
                    "family": family,
                    "candidate": "canal_validation_ranked",
                    "comparator": comparator,
                    "metric": metric,
                    "experimental_unit": "target",
                    "n_targets": int(len(delta)),
                    "delta_candidate_minus_comparator": float(delta.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "wilcoxon_p": p_value,
                    "target_wins": wins,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.result_dir

    candidate_files = [
        result_dir / "problem_check_chembl35_ic50_top20_v2_lgbm_ic50_lgbm_ecfp_desc_cw0p5_aw0p5_3seed.csv",
        result_dir / "problem_check_chembl35_ic50_top20_v2_lgbm_5seed_add_lgbm_ecfp_desc_cw0p5_aw0p5_s2069_2081.csv",
    ]
    final_files = [
        result_dir / "problem_check_chembl35_ic50_top20_v2_lgbm_ic50_lgbm_ecfp_desc_relation_cw0p25_aw1_right0p1_aw0_3seed.csv",
        result_dir / "problem_check_chembl35_ic50_top20_v2_lgbm_5seed_add_lgbm_ecfp_desc_relation_cw0p25_aw1_right0p1_aw0_s2069_2081.csv",
    ]
    matched_files = sorted(result_dir.glob("problem_check_chembl35_ic50_top20_v2_lgbm_ic50_lgbm_ecfp_desc_source_light_3seed_S?.csv"))
    matched_files.append(
        result_dir / "problem_check_chembl35_ic50_top20_v2_lgbm_5seed_add_lgbm_source_light_ecfp_desc_cw0p1_aw1_s2069_2081.csv"
    )

    candidate = read_many(candidate_files)
    candidate = candidate.loc[candidate["treatment"].eq("censored_likelihood")].copy()
    candidate["treatment"] = "canal_validation_ranked"
    final = read_many(final_files)
    final = final.loc[final["treatment"].eq("censored_likelihood")].copy()
    final["treatment"] = "canal_final"
    matched = read_many(matched_files)
    matched = matched.loc[matched["treatment"].eq("censored_as_exact")].copy()
    frame = pd.concat([matched, final, candidate], ignore_index=True)

    expected = {"censored_as_exact", "canal_final", "canal_validation_ranked"}
    if set(frame["treatment"].unique()) != expected:
        raise RuntimeError(f"Unexpected treatments: {sorted(frame['treatment'].unique())}")
    counts = frame.groupby("treatment").size()
    if not counts.eq(400).all():
        raise RuntimeError(f"Expected 400 rows per treatment, found {counts.to_dict()}")
    keys = ["target", "split", "seed"]
    for treatment, block in frame.groupby("treatment"):
        if block.duplicated(keys).any() or block[keys].drop_duplicates().shape[0] != 400:
            raise RuntimeError(f"Incomplete or duplicated cells for {treatment}")
        if block["status"].ne("ok").any():
            raise RuntimeError(f"Non-OK rows for {treatment}")

    summary_rows = []
    for treatment, block in frame.groupby("treatment"):
        for metric in METRICS:
            values = block[metric].dropna()
            summary_rows.append(
                {
                    "treatment": treatment,
                    "metric": metric,
                    "target_split_seed_cells": int(len(values)),
                    "mean": float(values.mean()),
                    "sd": float(values.std()),
                }
            )
    summary = pd.DataFrame(summary_rows)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    inference_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for comparator in ["censored_as_exact", "canal_final"]:
        inference_rows.extend(infer(frame, [], comparator, "aggregate_four_metrics", rng))
        split_rows.extend(infer(frame, ["split"], comparator, "four_splits_by_four_metrics", rng))
    inference = pd.DataFrame(inference_rows)
    inference["holm_p"] = inference.groupby("comparator", group_keys=False)["wilcoxon_p"].apply(holm_adjust)
    split_inference = pd.DataFrame(split_rows)
    split_inference["holm_p"] = split_inference.groupby("comparator", group_keys=False)["wilcoxon_p"].apply(holm_adjust)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "validation_ranked_primary_cells.csv", index=False)
    summary.to_csv(args.out_dir / "validation_ranked_primary_summary.csv", index=False)
    inference.to_csv(args.out_dir / "validation_ranked_primary_inference.csv", index=False)
    split_inference.to_csv(args.out_dir / "validation_ranked_primary_split_inference.csv", index=False)
    manifest = {
        "candidate_files": [str(path) for path in candidate_files],
        "final_files": [str(path) for path in final_files],
        "matched_files": [str(path) for path in matched_files],
        "candidate_hyperparameters": {
            "left_censor_weight": 0.5,
            "left_anchor_weight": 0.5,
            "right_censor_weight": 0.5,
            "right_anchor_weight": 0.5,
        },
        "selection_rule": "first-ranked candidate in the retrospective ChEMBL35 validation audit",
        "targets": 20,
        "splits": ["S1", "S2", "S3", "S4"],
        "seeds": [2027, 2039, 2053, 2069, 2081],
        "experimental_unit": "target",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    (args.out_dir / "validation_ranked_primary_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(inference.to_string(index=False))
    print(split_inference.to_string(index=False))


if __name__ == "__main__":
    main()
