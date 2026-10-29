#!/usr/bin/env python
"""Build split-preserving IC50 quality-filter subsets for Reviewer 2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subset_summary(name: str, frame: pd.DataFrame, path: Path) -> dict[str, object]:
    split_columns = [column for column in frame.columns if column.startswith("split_")]
    split_counts = {
        column: frame[column].value_counts(dropna=False).sort_index().to_dict()
        for column in split_columns
    }
    return {
        "name": name,
        "path": str(path),
        "sha256": sha256(path),
        "rows": int(len(frame)),
        "targets": int(frame["target_chembl_id"].nunique()),
        "exact_rows": int(frame["p_relation"].eq("=").sum()),
        "left_censored_rows": int(frame["p_relation"].eq("<").sum()),
        "right_censored_rows": int(frame["p_relation"].eq(">").sum()),
        "potential_duplicate_rows": int(frame["potential_duplicate"].fillna(0).astype(bool).sum()),
        "data_validity_comment_rows": int(frame["data_validity_comment"].notna().sum()),
        "split_counts": split_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    source = pd.read_parquet(args.dataset)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    duplicate = source["potential_duplicate"].fillna(0).astype(bool)
    validity = source["data_validity_comment"].notna()
    validity_text = source["data_validity_comment"].fillna("").astype(str)
    subsets = {
        "no_potential_duplicate": source.loc[~duplicate].copy(),
        "no_validity_comment": source.loc[~validity].copy(),
        "no_duplicate_or_validity_comment": source.loc[~duplicate & ~validity].copy(),
        "no_outside_typical_range": source.loc[
            ~validity_text.eq("Outside typical range")
        ].copy(),
        "no_potential_transcription_error": source.loc[
            ~validity_text.eq("Potential transcription error")
        ].copy(),
    }

    manifest: dict[str, object] = {
        "source": {
            "path": str(args.dataset),
            "sha256": sha256(args.dataset),
            "rows": int(len(source)),
        },
        "filters": {
            "no_potential_duplicate": "potential_duplicate is false or missing",
            "no_validity_comment": "data_validity_comment is missing",
            "no_duplicate_or_validity_comment": (
                "potential_duplicate is false or missing and data_validity_comment is missing"
            ),
            "no_outside_typical_range": (
                "data_validity_comment is not 'Outside typical range'"
            ),
            "no_potential_transcription_error": (
                "data_validity_comment is not 'Potential transcription error'"
            ),
        },
        "subsets": [],
    }
    for name, frame in subsets.items():
        output = args.out_dir / f"chembl35_ic50_top20_v2_{name}.parquet"
        frame.to_parquet(output, index=False)
        manifest["subsets"].append(subset_summary(name, frame, output))

    manifest_path = args.out_dir / "quality_subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
