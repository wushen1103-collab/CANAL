#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


SEEDS = [2027, 2039, 2053, 2069, 2081]
METRICS = ["test_mae", "test_censored_nll", "test_censored_dir_acc", "worst_assay_mae"]
BOOTSTRAP_SEED = 330041
BOOTSTRAP_REPLICATES = 20_000


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def summarize(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(groups, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, object] = dict(zip(groups, keys))
        row.update(runs=len(group), targets=group["target"].nunique(), seeds=group["seed"].nunique())
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def target_tests(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
    candidate_name: str,
    comparator_name: str,
    family: str,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    keys = ["target", "split", "seed"]
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        paired = candidate[keys + [metric]].merge(
            comparator[keys + [metric]], on=keys, suffixes=("_a", "_b"), validate="one_to_one"
        )
        if len(paired) != 400:
            raise RuntimeError(f"Expected 400 pairs for {candidate_name} vs {comparator_name}")
        delta = (
            paired.assign(delta=paired[f"{metric}_a"] - paired[f"{metric}_b"])
            .groupby("target")["delta"]
            .mean()
            .to_numpy(dtype=float)
        )
        sampled = delta[rng.integers(0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta)))].mean(axis=1)
        lower_is_better = metric != "test_censored_dir_acc"
        rows.append(
            {
                "family": family,
                "candidate": candidate_name,
                "comparator": comparator_name,
                "metric": metric,
                "experimental_unit": "target",
                "n_target_run_pairs": len(paired),
                "n_targets": len(delta),
                "delta_candidate_minus_comparator": delta.mean(),
                "ci_low": np.quantile(sampled, 0.025),
                "ci_high": np.quantile(sampled, 0.975),
                "p_raw": wilcoxon(delta).pvalue if np.any(delta != 0) else 1.0,
                "p_holm": np.nan,
                "wins": int((delta < 0).sum() if lower_is_better else (delta > 0).sum()),
            }
        )
    return rows


def holm(rows: list[dict[str, object]]) -> None:
    order = sorted(range(len(rows)), key=lambda index: float(rows[index]["p_raw"]))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * float(rows[index]["p_raw"])))
        rows[index]["p_holm"] = running
        rows[index]["holm_family_size"] = len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--tag", default="modern_src5")
    args = parser.parse_args()
    repo = args.repo.resolve()

    dmpnn_files = sorted((repo / "results" / "modern" / "chemprop").glob(f"*_{args.tag}.csv"))
    dmpnn_all = pd.concat(map(pd.read_csv, dmpnn_files), ignore_index=True)
    dmpnn = dmpnn_all[dmpnn_all["seed"].isin(SEEDS)].copy()
    attentive_files = sorted((repo / "results" / "modern" / "attentivefp").glob(f"*_{args.tag}.csv"))
    attentive = pd.concat(map(pd.read_csv, attentive_files), ignore_index=True)
    if len(dmpnn) != 1200 or len(attentive) != 1200:
        raise RuntimeError(f"Expected 1200 rows per neural route, found {len(dmpnn)} and {len(attentive)}")
    for name, frame in (("D-MPNN", dmpnn), ("AttentiveFP", attentive)):
        if not frame["status"].eq("ok").all():
            raise RuntimeError(f"Non-ok rows in {name}")
        if frame.duplicated(["target", "split", "seed", "treatment"]).any():
            raise RuntimeError(f"Duplicate rows in {name}")

    canal_files = [
        repo / "results" / "problem_check" / "problem_check_chembl35_ic50_top20_v2_lgbm_ic50_lgbm_ecfp_desc_relation_cw0p25_aw1_right0p1_aw0_3seed.csv",
        repo / "results" / "problem_check" / "problem_check_chembl35_ic50_top20_v2_lgbm_5seed_add_lgbm_ecfp_desc_relation_cw0p25_aw1_right0p1_aw0_s2069_2081.csv",
    ]
    canal = pd.concat(map(pd.read_csv, canal_files), ignore_index=True)
    canal["route"] = "CANAL"
    canal["display_method"] = "CANAL"
    dmpnn["route"] = "D-MPNN"
    attentive["route"] = "AttentiveFP"
    for frame in (dmpnn, attentive):
        frame["display_method"] = frame["route"] + " " + frame["treatment"]
    columns = ["target", "split", "seed", "route", "display_method", *METRICS]
    combined = pd.concat([canal[columns], dmpnn[columns], attentive[columns]], ignore_index=True)

    out_dir = repo / "reports" / "modern_neural"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "modern_neural_5seed_runs.csv", index=False)
    summarize(combined, ["display_method"]).to_csv(out_dir / "modern_neural_5seed_summary.csv", index=False)
    summarize(combined, ["display_method", "split"]).to_csv(out_dir / "modern_neural_5seed_by_split.csv", index=False)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    attentive_mechanism: list[dict[str, object]] = []
    attentive_bounded = attentive[attentive["treatment"].eq("bounded_mse")]
    for treatment in ("exact_only", "censored_as_exact"):
        attentive_mechanism.extend(
            target_tests(
                attentive_bounded,
                attentive[attentive["treatment"].eq(treatment)],
                "AttentiveFP bounded-MSE",
                f"AttentiveFP {treatment}",
                "attentivefp_treatment_8",
                rng,
            )
        )
    holm(attentive_mechanism)
    pd.DataFrame(attentive_mechanism).to_csv(out_dir / "attentivefp_treatment_target_clustered.csv", index=False)

    modern_tests: list[dict[str, object]] = []
    for route, frame in (("D-MPNN", dmpnn), ("AttentiveFP", attentive)):
        for treatment in ("bounded_mse", "censored_as_exact"):
            modern_tests.extend(
                target_tests(
                    canal,
                    frame[frame["treatment"].eq(treatment)],
                    "CANAL",
                    f"{route} {treatment}",
                    "canal_vs_modern_neural_16",
                    rng,
                )
            )
    holm(modern_tests)
    pd.DataFrame(modern_tests).to_csv(out_dir / "canal_vs_modern_neural_target_clustered.csv", index=False)

    attentive_runs = attentive.drop_duplicates(["split", "seed", "treatment"])
    convergence = attentive_runs.groupby("treatment")["best_epoch"].describe().reset_index()
    convergence["at_epoch_cap"] = convergence["treatment"].map(
        attentive_runs.assign(at_cap=attentive_runs["best_epoch"].eq(30)).groupby("treatment")["at_cap"].sum()
    )
    convergence.to_csv(out_dir / "attentivefp_convergence.csv", index=False)

    all_inputs = [*dmpnn_files, *attentive_files, *canal_files]
    graph_manifest = repo / "data" / "processed" / "attentivefp_graphs_ic50" / "manifest.json"
    provenance = {
        "formal_seeds": SEEDS,
        "rows": int(len(combined)),
        "methods": sorted(combined["display_method"].unique().tolist()),
        "input_sha256": {str(path.relative_to(repo)): digest(path) for path in [*all_inputs, graph_manifest]},
        "attentivefp_architecture": {"hidden_channels": 200, "layers": 2, "timesteps": 2, "dropout": 0.2},
        "attentivefp_training": {"epochs": 30, "patience": 6, "batch_size": 512, "source_light": True},
        "graph_manifest": json.loads(graph_manifest.read_text(encoding="utf-8")),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "holm_family_size": len(modern_tests),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    print(summarize(combined, ["display_method"]).to_string(index=False))
    print(pd.DataFrame(modern_tests).to_string(index=False))


if __name__ == "__main__":
    main()
