#!/usr/bin/env python
"""Summarize final-backbone component ablations with target-level inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


BOOTSTRAP_SEED = 360_033
BOOTSTRAP_REPLICATES = 20_000
KEYS = ["target", "split", "seed"]
METRICS = {
    "test_mae": "lower",
    "test_censored_nll": "lower",
    "test_censored_dir_acc": "higher",
    "worst_assay_mae": "lower",
}
CONFIG_PATHS = {
    "Full CANAL": [
        "results/problem_check/problem_check_chembl35_ic50_top20_v2_lgbm_ic50_lgbm_ecfp_desc_relation_cw0p25_aw1_right0p1_aw0_3seed.csv",
        "results/problem_check/problem_check_chembl35_ic50_top20_v2_lgbm_5seed_add_lgbm_ecfp_desc_relation_cw0p25_aw1_right0p1_aw0_s2069_2081.csv",
    ],
    "No source metadata": [
        "results/problem_check/problem_check_chembl35_ic50_top20_v2_lgbm_ablation_final_no_source_5seed_20260903.csv",
    ],
    "No threshold anchor": [
        "results/problem_check/problem_check_chembl35_ic50_top20_v2_lgbm_ablation_final_no_anchor_5seed_20260903.csv",
    ],
    "Relation-symmetric": [
        "results/problem_check/problem_check_chembl35_ic50_top20_v2_lgbm_ic50_lgbm_ecfp_desc_cw0p25_aw1_3seed.csv",
        "results/problem_check/problem_check_chembl35_ic50_top20_v2_lgbm_5seed_add_lgbm_ecfp_desc_cw0p25_aw1_s2069_2081.csv",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holm_adjust(frame: pd.DataFrame) -> pd.Series:
    values = frame["p_raw"].to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return pd.Series(adjusted, index=frame.index)


def load_config(repo: Path, label: str, relative_paths: list[str]) -> pd.DataFrame:
    paths = [repo / path for path in relative_paths]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {label} files: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame = frame.loc[frame["status"].eq("ok")].copy()
    if len(frame) != 400 or frame["target"].nunique() != 20 or frame["seed"].nunique() != 5:
        raise RuntimeError(
            f"{label}: expected 400 valid rows, 20 targets, and five seeds; "
            f"observed rows={len(frame)}, targets={frame['target'].nunique()}, seeds={frame['seed'].nunique()}"
        )
    if frame.duplicated(KEYS).any():
        raise RuntimeError(f"{label}: duplicate target-split-seed rows")
    frame["configuration"] = label
    return frame


def bootstrap_mean(delta: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta)))
    sampled = delta[indices].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evidence_completion_2026-09-03/matched_ablation"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output if args.output.is_absolute() else repo / args.output
    output.mkdir(parents=True, exist_ok=True)

    configs = {label: load_config(repo, label, paths) for label, paths in CONFIG_PATHS.items()}
    combined = pd.concat(configs.values(), ignore_index=True)
    summary_rows = []
    for label, frame in configs.items():
        row: dict[str, object] = {
            "configuration": label,
            "runs": len(frame),
            "targets": frame["target"].nunique(),
            "splits": frame["split"].nunique(),
            "seeds": frame["seed"].nunique(),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = frame[metric].mean()
            row[f"{metric}_std"] = frame[metric].std(ddof=1)
            row[f"{metric}_n"] = frame[metric].notna().sum()
        summary_rows.append(row)

    full = configs["Full CANAL"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    effect_rows = []
    target_rows = []
    for label, comparator in configs.items():
        if label == "Full CANAL":
            continue
        for metric, direction in METRICS.items():
            full_target = full.groupby("target")[metric].mean().rename("full")
            comparator_target = comparator.groupby("target")[metric].mean().rename("comparator")
            paired = pd.concat([full_target, comparator_target], axis=1).dropna()
            paired["delta_full_minus_comparator"] = paired["full"] - paired["comparator"]
            delta = paired["delta_full_minus_comparator"].to_numpy(dtype=float)
            low, high = bootstrap_mean(delta, rng)
            try:
                p_raw = float(wilcoxon(delta).pvalue)
            except ValueError:
                p_raw = 1.0
            favorable = delta < 0 if direction == "lower" else delta > 0
            effect_rows.append(
                {
                    "comparator": label,
                    "metric": metric,
                    "experimental_unit": "target",
                    "n_targets": len(delta),
                    "delta_full_minus_comparator": delta.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "p_raw": p_raw,
                    "wins": int(favorable.sum()),
                    "holm_family_size": (len(configs) - 1) * len(METRICS),
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
            for target, values in paired.iterrows():
                target_rows.append(
                    {
                        "comparator": label,
                        "metric": metric,
                        "target": target,
                        "full": values["full"],
                        "comparator_value": values["comparator"],
                        "delta_full_minus_comparator": values["delta_full_minus_comparator"],
                    }
                )

    effects = pd.DataFrame(effect_rows)
    effects["p_holm"] = holm_adjust(effects)
    paths = {
        "summary": output.with_name(output.name + "_summary.csv"),
        "tests": output.with_name(output.name + "_tests.csv"),
        "target_effects": output.with_name(output.name + "_target_effects.csv"),
        "runs": output.with_name(output.name + "_runs.csv"),
        "manifest": output.with_name(output.name + "_manifest.json"),
    }
    pd.DataFrame(summary_rows).to_csv(paths["summary"], index=False)
    effects.to_csv(paths["tests"], index=False)
    pd.DataFrame(target_rows).to_csv(paths["target_effects"], index=False)
    combined.to_csv(paths["runs"], index=False)
    manifest = {
        "configurations": CONFIG_PATHS,
        "source_sha256": {
            path: sha256(repo / path)
            for paths in CONFIG_PATHS.values()
            for path in paths
        },
        "experimental_unit": "target after averaging split and seed",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "holm_family_size": int(len(effects)),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(paths["summary"])
    print(paths["tests"])


if __name__ == "__main__":
    main()
