#!/usr/bin/env python3
"""Tabulate how component closure changes the five-seed S5 conclusions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "test_mae",
    "test_censored_nll",
    "test_censored_dir_acc",
    "worst_assay_mae",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-prefix", type=Path, required=True)
    parser.add_argument("--new-prefix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for protocol, prefix in (("original", args.old_prefix), ("component_closed", args.new_prefix)):
        frame = pd.read_csv(prefix.with_name(prefix.name + "_summary.csv"))
        for _, row in frame.iterrows():
            for metric in METRICS:
                summaries.append(
                    {
                        "protocol": protocol,
                        "configuration": row["configuration"],
                        "metric": metric,
                        "mean": row[f"{metric}_mean"],
                        "std": row[f"{metric}_std"],
                        "runs": row["runs"],
                    }
                )
    summary = pd.DataFrame(summaries)
    wide = summary.pivot(index=["configuration", "metric"], columns="protocol", values="mean")
    wide["component_closed_minus_original"] = wide["component_closed"] - wide["original"]
    wide.reset_index().to_csv(args.output_dir / "r2_3_s5_protocol_result_change.csv", index=False)
    summary.to_csv(args.output_dir / "r2_3_s5_protocol_result_long.csv", index=False)

    tests = []
    for protocol, prefix in (("original", args.old_prefix), ("component_closed", args.new_prefix)):
        frame = pd.read_csv(prefix.with_name(prefix.name + "_tests.csv"))
        frame.insert(0, "protocol", protocol)
        tests.append(frame)
    pd.concat(tests, ignore_index=True).to_csv(
        args.output_dir / "r2_3_s5_protocol_inference_change.csv", index=False
    )


if __name__ == "__main__":
    main()
