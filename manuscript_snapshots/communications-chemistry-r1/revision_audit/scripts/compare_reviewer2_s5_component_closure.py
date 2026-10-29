#!/usr/bin/env python3
"""Compare the original and component-closed S5 assignments."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--closed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original = pd.read_parquet(args.original)
    closed = pd.read_parquet(args.closed)
    if len(original) != len(closed):
        raise ValueError("S5 datasets have different row counts")
    identity_columns = ["target_chembl_id", "parent_inchikey", "document_chembl_id", "p_activity", "p_relation"]
    if not original[identity_columns].equals(closed[identity_columns]):
        raise ValueError("S5 closure changed row identity or response content")

    comparison = pd.DataFrame(
        {
            "target": original["target_chembl_id"],
            "original": original["split_s5"],
            "closed": closed["split_s5"],
        }
    )
    crosstab = pd.crosstab(comparison["original"], comparison["closed"], dropna=False)
    crosstab.to_csv(args.output_dir / "r2_13_s5_original_vs_component_closed_crosstab.csv")
    per_target = (
        comparison.assign(changed=comparison["original"].ne(comparison["closed"]))
        .groupby("target", as_index=False)
        .agg(rows=("target", "size"), changed_rows=("changed", "sum"))
    )
    for label, frame in (("original", original), ("closed", closed)):
        counts = frame.groupby(["target_chembl_id", "split_s5"]).size().unstack(fill_value=0)
        counts = counts.reindex(per_target["target"]).reset_index(drop=True)
        for partition in ("train", "val", "test", "exclude"):
            per_target[f"{label}_{partition}"] = counts.get(partition, 0)
    per_target.to_csv(args.output_dir / "r2_13_s5_component_closure_by_target.csv", index=False)

    test_unchanged = original["split_s5"].eq("test").equals(closed["split_s5"].eq("test"))
    manifest = {
        "rows": int(len(original)),
        "changed_rows": int(comparison["original"].ne(comparison["closed"]).sum()),
        "changed_targets": int(per_target["changed_rows"].gt(0).sum()),
        "test_membership_unchanged": bool(test_unchanged),
        "response_values_unchanged": True,
        "original_sha256": sha256(args.original),
        "component_closed_sha256": sha256(args.closed),
        "crosstab": {
            str(row): {str(column): int(value) for column, value in values.items()}
            for row, values in crosstab.to_dict(orient="index").items()
        },
    }
    (args.output_dir / "r2_13_s5_component_closure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
