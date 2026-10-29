#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


BOOTSTRAP_SEED = 360033
BOOTSTRAP_REPLICATES = 20_000
METRICS = ["test_mae", "test_censored_nll", "test_censored_dir_acc", "worst_assay_mae"]
COMPARATORS = ["exact_only", "censored_as_exact"]


def holm_adjust(rows: list[dict[str, object]]) -> None:
    valid = [i for i, row in enumerate(rows) if np.isfinite(float(row["p_raw"]))]
    order = sorted(valid, key=lambda i: float(rows[i]["p_raw"]))
    running_max = 0.0
    for rank, row_index in enumerate(order):
        candidate = min(1.0, (len(order) - rank) * float(rows[row_index]["p_raw"]))
        running_max = max(running_max, candidate)
        rows[row_index]["p_holm"] = running_max


def cohort_name(dataset: str) -> str:
    return "novel_parent" if str(dataset).endswith("_novel_parent") else "release_update"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--tag", default="locked5")
    args = parser.parse_args()
    repo = args.repo.resolve()
    files = sorted(
        (repo / "results" / "problem_check").glob(
            f"problem_check_chembl36_ic50_release_update*_lgbm_*{args.tag}.csv"
        )
    )
    if len(files) != 8:
        raise RuntimeError(f"Expected eight batch files, found {len(files)}: {files}")

    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    frame["cohort"] = frame["dataset"].map(cohort_name)
    expected_rows = 2 * 20 * 5 * 3
    if len(frame) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, found {len(frame)}")
    if not frame["status"].eq("ok").all():
        raise RuntimeError(str(frame.loc[frame["status"].ne("ok"), ["cohort", "target", "seed", "status"]]))
    if frame.duplicated(["cohort", "target", "seed", "treatment"]).any():
        raise RuntimeError("Duplicate cohort-target-seed-treatment rows found")

    candidate = frame[frame["treatment"].eq("censored_likelihood")]
    checks = {
        "censor_weight": 0.25,
        "censored_anchor_weight": 1.0,
        "right_censor_weight": 0.1,
        "right_censored_anchor_weight": 0.0,
    }
    for column, expected in checks.items():
        if not np.allclose(candidate[column].astype(float), expected):
            raise RuntimeError(f"Frozen CANAL setting mismatch in {column}")

    out_dir = repo / "reports" / "chembl36_release_update"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "chembl36_release_update_5seed_runs.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    for (cohort, treatment), group in frame.groupby(["cohort", "treatment"], sort=True):
        row: dict[str, object] = {
            "cohort": cohort,
            "treatment": treatment,
            "runs": int(len(group)),
            "targets": int(group["target"].nunique()),
            "seeds": int(group["seed"].nunique()),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_n"] = int(values.notna().sum())
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "chembl36_release_update_5seed_summary.csv", index=False)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    inference_rows: list[dict[str, object]] = []
    for cohort in ["release_update", "novel_parent"]:
        cohort_frame = frame[frame["cohort"].eq(cohort)]
        main = cohort_frame[cohort_frame["treatment"].eq("censored_likelihood")]
        for comparator in COMPARATORS:
            baseline = cohort_frame[cohort_frame["treatment"].eq(comparator)]
            for metric in METRICS:
                main_target = main.groupby("target")[metric].mean().rename("main")
                base_target = baseline.groupby("target")[metric].mean().rename("baseline")
                paired = pd.concat([main_target, base_target], axis=1).dropna()
                delta = (paired["main"] - paired["baseline"]).to_numpy(dtype=float)
                if len(delta):
                    indices = rng.integers(0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta)))
                    bootstrap_mean = delta[indices].mean(axis=1)
                    p_raw = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
                    ci_low = float(np.quantile(bootstrap_mean, 0.025))
                    ci_high = float(np.quantile(bootstrap_mean, 0.975))
                else:
                    p_raw = ci_low = ci_high = float("nan")
                lower_is_better = metric != "test_censored_dir_acc"
                inference_rows.append(
                    {
                        "cohort": cohort,
                        "comparator": comparator,
                        "metric": metric,
                        "experimental_unit": "target",
                        "n_targets": int(len(delta)),
                        "delta_canal_minus_comparator": float(delta.mean()) if len(delta) else float("nan"),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_raw": p_raw,
                        "p_holm": float("nan"),
                        "wins": int((delta < 0).sum() if lower_is_better else (delta > 0).sum()),
                        "bootstrap_seed": BOOTSTRAP_SEED,
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    }
                )
    holm_adjust(inference_rows)
    inference = pd.DataFrame(inference_rows)
    inference["holm_family_size"] = int(inference["p_raw"].notna().sum())
    inference.to_csv(out_dir / "chembl36_release_update_target_clustered.csv", index=False)

    provenance = {
        "input_files": {str(path.relative_to(repo)): path.stat().st_size for path in files},
        "rows": int(len(frame)),
        "status_counts": {str(k): int(v) for k, v in frame["status"].value_counts().items()},
        "cohorts": sorted(frame["cohort"].unique().tolist()),
        "targets_per_cohort": {str(k): int(v) for k, v in frame.groupby("cohort")["target"].nunique().items()},
        "seeds": sorted(int(x) for x in frame["seed"].unique()),
        "treatments": sorted(frame["treatment"].unique().tolist()),
        "canal_setting": checks,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "holm_family_size": int(inference["p_raw"].notna().sum()),
    }
    (out_dir / "chembl36_release_update_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(inference.to_string(index=False))


if __name__ == "__main__":
    main()
