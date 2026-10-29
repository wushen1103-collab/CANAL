#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


SEEDS = [2027, 2039, 2053, 2069, 2081]
METRICS = ["test_mae", "test_censored_nll", "test_censored_dir_acc", "worst_assay_mae"]
BOOTSTRAP_SEED = 330039
BOOTSTRAP_REPLICATES = 20_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def holm_adjust(rows: list[dict[str, object]]) -> None:
    valid = [index for index, row in enumerate(rows) if np.isfinite(float(row["p_raw"]))]
    order = sorted(valid, key=lambda index: float(rows[index]["p_raw"]))
    running_max = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(order) - rank) * float(rows[index]["p_raw"]))
        running_max = max(running_max, adjusted)
        rows[index]["p_holm"] = running_max


def summarize(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(groups, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, object] = dict(zip(groups, keys))
        row.update(
            runs=int(len(group)),
            targets=int(group["target"].nunique()),
            seeds=int(group["seed"].nunique()),
        )
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def target_clustered(
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
            comparator[keys + [metric]], on=keys, suffixes=("_candidate", "_comparator"), validate="one_to_one"
        )
        if len(paired) != 400:
            raise RuntimeError(f"Expected 400 pairs for {candidate_name} versus {comparator_name}, found {len(paired)}")
        paired["delta"] = paired[f"{metric}_candidate"] - paired[f"{metric}_comparator"]
        delta = paired.groupby("target")["delta"].mean().to_numpy(dtype=float)
        indices = rng.integers(0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta)))
        boot = delta[indices].mean(axis=1)
        p_raw = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
        lower_is_better = metric != "test_censored_dir_acc"
        rows.append(
            {
                "family": family,
                "candidate": candidate_name,
                "comparator": comparator_name,
                "metric": metric,
                "experimental_unit": "target",
                "n_target_run_pairs": int(len(paired)),
                "n_targets": int(len(delta)),
                "delta_candidate_minus_comparator": float(delta.mean()),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "p_raw": p_raw,
                "p_holm": float("nan"),
                "wins": int((delta < 0).sum() if lower_is_better else (delta > 0).sum()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--tag", default="modern_src5")
    args = parser.parse_args()
    repo = args.repo.resolve()
    files = sorted((repo / "results" / "modern" / "chemprop").glob(f"*_{args.tag}.csv"))
    all_dmpnn = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    dmpnn = all_dmpnn[all_dmpnn["seed"].isin(SEEDS)].copy()
    if len(files) != 72 or len(all_dmpnn) != 1440:
        raise RuntimeError(f"Expected 72 files/1440 rows including seed 2041, found {len(files)}/{len(all_dmpnn)}")
    if len(dmpnn) != 1200 or not dmpnn["status"].eq("ok").all():
        raise RuntimeError("Incomplete five-seed Chemprop matrix")
    if dmpnn.duplicated(["target", "split", "seed", "treatment"]).any():
        raise RuntimeError("Duplicate Chemprop target-split-seed-treatment rows")
    if set(dmpnn["epochs_requested"].astype(int)) != {30}:
        raise RuntimeError("Unexpected Chemprop epoch budget")

    canal_files = [
        repo / "results" / "problem_check" / "problem_check_chembl35_ic50_top20_v2_lgbm_ic50_lgbm_ecfp_desc_relation_cw0p25_aw1_right0p1_aw0_3seed.csv",
        repo / "results" / "problem_check" / "problem_check_chembl35_ic50_top20_v2_lgbm_5seed_add_lgbm_ecfp_desc_relation_cw0p25_aw1_right0p1_aw0_s2069_2081.csv",
    ]
    canal = pd.concat([pd.read_csv(path) for path in canal_files], ignore_index=True)
    if len(canal) != 400 or set(canal["seed"].astype(int)) != set(SEEDS):
        raise RuntimeError("Incomplete or seed-mismatched CANAL comparison rows")
    checks = {
        "censor_weight": 0.25,
        "censored_anchor_weight": 1.0,
        "right_censor_weight": 0.1,
        "right_censored_anchor_weight": 0.0,
    }
    for column, expected in checks.items():
        if not np.allclose(canal[column].astype(float), expected):
            raise RuntimeError(f"CANAL setting mismatch: {column}")

    out_dir = repo / "reports" / "modern_chemprop"
    out_dir.mkdir(parents=True, exist_ok=True)
    dmpnn.to_csv(out_dir / "chemprop_dmpnn_5seed_runs.csv", index=False)
    summarize(dmpnn, ["treatment"]).to_csv(out_dir / "chemprop_dmpnn_5seed_summary.csv", index=False)
    summarize(dmpnn, ["treatment", "split"]).to_csv(out_dir / "chemprop_dmpnn_5seed_by_split.csv", index=False)

    canal_standard = canal.rename(columns={"test_censored_nll": "test_censored_nll"}).copy()
    canal_standard["treatment"] = "CANAL"
    combined = pd.concat(
        [canal_standard[["target", "split", "seed", "treatment", *METRICS]], dmpnn[["target", "split", "seed", "treatment", *METRICS]]],
        ignore_index=True,
    )
    summarize(combined, ["treatment"]).to_csv(out_dir / "canal_vs_dmpnn_summary.csv", index=False)
    summarize(combined, ["treatment", "split"]).to_csv(out_dir / "canal_vs_dmpnn_by_split.csv", index=False)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    mechanism_rows: list[dict[str, object]] = []
    bounded = dmpnn[dmpnn["treatment"].eq("bounded_mse")]
    for comparator in ("exact_only", "censored_as_exact"):
        mechanism_rows.extend(
            target_clustered(
                bounded,
                dmpnn[dmpnn["treatment"].eq(comparator)],
                "D-MPNN bounded-MSE",
                f"D-MPNN {comparator}",
                "dmpnn_treatment_8",
                rng,
            )
        )
    holm_adjust(mechanism_rows)
    for row in mechanism_rows:
        row["holm_family_size"] = len(mechanism_rows)
    pd.DataFrame(mechanism_rows).to_csv(out_dir / "dmpnn_treatment_target_clustered.csv", index=False)

    sota_rows: list[dict[str, object]] = []
    for comparator in ("bounded_mse", "censored_as_exact"):
        sota_rows.extend(
            target_clustered(
                canal_standard,
                dmpnn[dmpnn["treatment"].eq(comparator)],
                "CANAL",
                f"D-MPNN {comparator}",
                "canal_vs_modern_8",
                rng,
            )
        )
    holm_adjust(sota_rows)
    for row in sota_rows:
        row["holm_family_size"] = len(sota_rows)
    pd.DataFrame(sota_rows).to_csv(out_dir / "canal_vs_dmpnn_target_clustered.csv", index=False)

    extra_seed = summarize(all_dmpnn[all_dmpnn["seed"].eq(2041)], ["treatment"])
    extra_seed.to_csv(out_dir / "chemprop_seed2041_sensitivity.csv", index=False)
    manifest_path = repo / "data" / "processed" / "chemprop_multitask_ic50" / "manifest.json"
    provenance = {
        "formal_seeds": SEEDS,
        "extra_sensitivity_seed": 2041,
        "formal_rows": int(len(dmpnn)),
        "formal_files": len([path for path in files if any(f"s{seed}_" in path.name for seed in SEEDS)]),
        "input_sha256": {str(path.relative_to(repo)): sha256(path) for path in [*files, *canal_files, manifest_path]},
        "versions": {name: version(name) for name in ["chemprop", "torch", "torch-geometric", "rdkit", "pandas", "scipy"]},
        "architecture": "Chemprop 2.3.1 default D-MPNN, depth 3, hidden size 300",
        "training": {"epochs": 30, "patience": 6, "batch_size": 512, "source_light": True},
        "source_feature_manifest": json.loads(manifest_path.read_text(encoding="utf-8"))["source_light"],
        "canal_setting": checks,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")

    print(summarize(combined, ["treatment"]).to_string(index=False))
    print(pd.DataFrame(mechanism_rows).to_string(index=False))
    print(pd.DataFrame(sota_rows).to_string(index=False))


if __name__ == "__main__":
    main()
