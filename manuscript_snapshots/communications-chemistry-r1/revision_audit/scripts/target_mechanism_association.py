#!/usr/bin/env python
"""Relate target-level CANAL gains to censoring and shift diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BOOTSTRAP_SEED = 330033
BOOTSTRAP_REPLICATES = 20_000
PRIMARY_SPLITS = {"S1", "S2", "S3", "S4"}


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & weights.gt(0)
    return float(np.average(values.loc[mask], weights=weights.loc[mask]))


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, position in enumerate(order):
        running = max(running, (total - rank) * values[position])
        adjusted[position] = min(running, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def bootstrap_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    valid = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        index = rng.integers(0, len(x), len(x))
        rho = spearmanr(x[index], y[index]).statistic
        if np.isfinite(rho):
            estimates[valid] = rho
            valid += 1
    if valid == 0:
        return np.nan, np.nan
    return tuple(np.quantile(estimates[:valid], [0.025, 0.975]))


def aggregate_primary_covariates(root: Path, target_effects: Path) -> pd.DataFrame:
    frame = pd.read_parquet(root / "data/processed/chembl35_ic50_top20_v2.parquet")
    relation = frame["p_relation"].astype(str)
    censoring = (
        frame.assign(
            is_censored=relation.ne("=").astype(int),
            is_left=relation.eq("<").astype(int),
        )
        .groupby("target_chembl_id", as_index=False)
        .agg(rows=("p_relation", "size"), censored_rows=("is_censored", "sum"), left_rows=("is_left", "sum"))
    )
    censoring["censored_fraction"] = censoring["censored_rows"] / censoring["rows"]
    censoring["left_share_among_censored"] = censoring["left_rows"] / censoring["censored_rows"]

    exact = frame.loc[relation.eq("="), ["target_chembl_id", "parent_inchikey", "p_activity"]].dropna()
    repeated = (
        exact.groupby(["target_chembl_id", "parent_inchikey"])["p_activity"]
        .agg(["size", "min", "max"])
        .reset_index()
    )
    repeated = repeated.loc[repeated["size"].gt(1)].copy()
    repeated["range"] = repeated["max"] - repeated["min"]
    dispersion = (
        repeated.groupby("target_chembl_id", as_index=False)
        .agg(repeat_groups=("range", "size"), repeat_range_median=("range", "median"))
    )

    similarity = pd.read_csv(
        root / "reports/similarity_diagnostics_2026-09-03/applicability_domain_by_target.csv"
    )
    similarity = similarity.loc[similarity["split"].isin(PRIMARY_SPLITS)].copy()
    similarity = (
        similarity.groupby("target_chembl_id")
        .apply(lambda group: weighted_mean(group["median"], group["test_molecules"]), include_groups=False)
        .rename("nearest_train_similarity")
        .reset_index()
    )

    source_shift = pd.read_csv(root / "reports/evidence_completion_2026-09-03/source_shift_by_target.csv")
    source_shift = source_shift.loc[
        source_shift["endpoint"].eq("IC50") & source_shift["split"].isin(PRIMARY_SPLITS)
    ].copy()
    js_columns = [
        "assay_type_js_divergence",
        "assay_organism_js_divergence",
        "confidence_score_js_divergence",
    ]
    source_shift["source_shift_js"] = source_shift[js_columns].mean(axis=1)
    source_shift = (
        source_shift.groupby("target", as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "source_shift_js": weighted_mean(group["source_shift_js"], group["test_rows"]),
                }
            ),
            include_groups=False,
        )
        .rename(columns={"target": "target_chembl_id"})
    )

    effects = pd.read_csv(target_effects).rename(columns={"target": "target_chembl_id"})
    effects["nll_gain"] = -effects["delta_test_censored_nll"]
    effects["cdir_gain"] = effects["delta_test_censored_dir_acc"]
    return (
        effects.merge(censoring, on="target_chembl_id", validate="one_to_one")
        .merge(dispersion, on="target_chembl_id", validate="one_to_one")
        .merge(similarity, on="target_chembl_id", validate="one_to_one")
        .merge(source_shift, on="target_chembl_id", validate="one_to_one")
    )


def aggregate_s5_covariates(
    diagnostics_path: Path,
    runs_path: Path,
) -> pd.DataFrame:
    diagnostics = pd.read_csv(diagnostics_path)
    scores = diagnostics.pivot(index="target", columns="split", values="source_tail_score_mean")
    scores["s5_source_context_distance"] = scores["test"] - scores["train"]

    runs = pd.read_csv(runs_path)
    runs = runs.loc[
        runs["status"].eq("ok")
        & runs["model"].eq("lgbm")
        & runs["metadata_mode"].eq("source_light")
        & runs["treatment"].isin(["censored_as_exact", "censored_likelihood"])
    ].copy()
    metric_columns = ["test_censored_nll", "test_censored_dir_acc"]
    means = runs.groupby(["target", "treatment"])[metric_columns].mean().unstack("treatment")
    output = pd.DataFrame(index=means.index)
    output["s5_nll_gain"] = (
        means[("test_censored_nll", "censored_as_exact")]
        - means[("test_censored_nll", "censored_likelihood")]
    )
    output["s5_cdir_gain"] = (
        means[("test_censored_dir_acc", "censored_likelihood")]
        - means[("test_censored_dir_acc", "censored_as_exact")]
    )
    return output.join(scores[["s5_source_context_distance"]]).reset_index()


def association_rows(
    frame: pd.DataFrame,
    family: str,
    predictors: list[str],
    outcomes: list[str],
) -> pd.DataFrame:
    rows = []
    for predictor in predictors:
        for outcome in outcomes:
            complete = frame[[predictor, outcome]].dropna()
            x = complete[predictor].to_numpy(dtype=float)
            y = complete[outcome].to_numpy(dtype=float)
            result = spearmanr(x, y)
            ci_low, ci_high = bootstrap_spearman(x, y)
            rows.append(
                {
                    "family": family,
                    "predictor": predictor,
                    "outcome": outcome,
                    "targets": len(complete),
                    "spearman_rho": float(result.statistic),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_raw": float(result.pvalue),
                }
            )
    output = pd.DataFrame(rows)
    output["p_holm"] = holm_adjust(output["p_raw"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--target-effects",
        type=Path,
        default=Path("reports/revision_target_effects_input.csv"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("reports/target_mechanism_association"),
    )
    parser.add_argument(
        "--s5-diagnostics",
        type=Path,
        default=Path("reports/evidence_completion_2026-09-03/source_context_s5_diagnostics.csv"),
    )
    parser.add_argument(
        "--s5-runs",
        type=Path,
        default=Path("reports/evidence_completion_2026-09-03/source_context_s5_results_runs.csv"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    target_effects = args.target_effects
    if not target_effects.is_absolute():
        target_effects = root / target_effects

    s5_diagnostics = args.s5_diagnostics
    if not s5_diagnostics.is_absolute():
        s5_diagnostics = root / s5_diagnostics
    s5_runs = args.s5_runs
    if not s5_runs.is_absolute():
        s5_runs = root / s5_runs

    primary = aggregate_primary_covariates(root, target_effects)
    s5 = aggregate_s5_covariates(s5_diagnostics, s5_runs)
    primary_results = association_rows(
        primary,
        "primary_S1-S4",
        [
            "censored_fraction",
            "left_share_among_censored",
            "repeat_range_median",
            "nearest_train_similarity",
            "source_shift_js",
        ],
        ["nll_gain", "cdir_gain"],
    )
    s5_results = association_rows(
        s5,
        "S5_source_context_tail",
        ["s5_source_context_distance"],
        ["s5_nll_gain", "s5_cdir_gain"],
    )
    results = pd.concat([primary_results, s5_results], ignore_index=True)

    prefix = args.output_prefix
    if not prefix.is_absolute():
        prefix = root / prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    primary.to_csv(prefix.with_name(prefix.name + "_target_covariates.csv"), index=False)
    s5.to_csv(prefix.with_name(prefix.name + "_s5_target_covariates.csv"), index=False)
    results.to_csv(prefix.with_name(prefix.name + "_results.csv"), index=False)
    manifest = {
        "analysis_unit": "target",
        "primary_splits": sorted(PRIMARY_SPLITS),
        "gain_orientation": "positive values favor CANAL",
        "association": "two-sided Spearman rank correlation",
        "confidence_interval": f"percentile target bootstrap, {BOOTSTRAP_REPLICATES} replicates",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "multiplicity": "Holm adjustment separately within the 10 primary and 2 S5 tests",
        "causal_interpretation": False,
        "s5_diagnostics": str(s5_diagnostics),
        "s5_runs": str(s5_runs),
    }
    prefix.with_name(prefix.name + "_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
