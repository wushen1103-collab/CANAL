#!/usr/bin/env python3
"""Verify the compact result snapshot distributed with this repository."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".yml", ".yaml", ".cff"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def portable_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore", ".gitattributes"}:
        data = data.replace(b"\r\n", b"\n")
    return data


def verify_checksums() -> None:
    manifest = ROOT / "release_checksums.sha256"
    if not manifest.exists():
        return
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        observed = hashlib.sha256(portable_bytes(path)).hexdigest()
        require(observed == expected, f"checksum {relative}")


def main() -> None:
    complete = pd.read_csv(ROOT / "results/benchmark/metrics.csv")
    strict = pd.read_csv(ROOT / "results/external_strict/metrics.csv")
    paired = pd.read_csv(ROOT / "results/benchmark/paired_statistics.csv")

    require(len(complete) == 1350, "complete benchmark has 1,350 aggregate metric records")
    semisynthetic = complete["semisynthetic"].astype(str).str.lower().eq("true")
    primary_experiments = {"core", "tree_baselines", "canal_calibrated_backbone", "smiles_cnn_baseline"}
    chembl = complete[
        (complete["dataset"] == "ChEMBL37-CANAL20")
        & complete["experiment"].isin(primary_experiments)
        & (~semisynthetic)
    ]
    require(len(chembl) == 375, "primary ChEMBL grid has 15 methods x 5 splits x 5 seeds")
    require(chembl["method"].nunique() == 15, "primary ChEMBL grid contains 15 method variants")
    require(len(strict) == 175, "strict external grid has 7 methods x 5 splits x 5 seeds")

    key = ["split", "seed"]
    primary = chembl[chembl["method"] == "canal_extratrees_molecule_calibrated"][key + ["spearman_exact"]]
    baseline = chembl[chembl["method"] == "extratrees_molecule_only_naive"][key + ["spearman_exact"]]
    matched = primary.merge(baseline, on=key, suffixes=("_canal", "_baseline"))
    require(len(matched) == 25, "rank-preservation audit has 25 matched split-seed evaluations")
    require(np.allclose(matched["spearman_exact_canal"], matched["spearman_exact_baseline"], atol=1e-12), "positive affine calibration leaves task-macro Spearman unchanged")

    require(len(paired) == 18, "paired analysis contains 18 prespecified tests")
    require(set(paired["n_pairs"]) == {20}, "every inferential comparison uses 20 target-level paired units")

    for name in ["chembl37", "bindingdb_202608"]:
        identity = pd.read_csv(ROOT / f"manifests/{name}_task_identity.csv")
        require(len(identity) == 20 and identity["one_hot_index"].nunique() == 20, f"{name} uses 20 collision-free one-hot task identities")

    overlap = pd.read_csv(ROOT / "results/audits/bindingdb_overlap_summary.csv").set_index("flag")
    require(int(overlap.loc["overlap_article", "rows"]) == 291, "article-overlap audit reports 291 rows")
    require(int(overlap.loc["overlap_canonical_molecule_target", "rows"]) == 14, "canonical molecule-target audit reports 14 rows")
    require(int(overlap.loc["overlap_exact_molecule_target_endpoint", "rows"]) == 0, "exact molecule-target-endpoint overlap is zero")

    verify_checksums()


if __name__ == "__main__":
    main()
