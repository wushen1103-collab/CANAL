#!/usr/bin/env python3
"""Summarize observed pActivity and censored-threshold ranges by endpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ENDPOINTS = ("EC50", "IC50", "Kd", "Ki")


def finite_range(values: pd.Series) -> tuple[float, float]:
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    if finite.empty:
        return float("nan"), float("nan")
    return float(finite.min()), float(finite.max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    for endpoint in ENDPOINTS:
        path = args.data_dir / f"chembl35_{endpoint}_top20_v2.parquet"
        frame = pd.read_parquet(path, columns=["p_activity", "is_exact"])
        frame = frame.loc[np.isfinite(frame["p_activity"].to_numpy(dtype=float))].copy()
        exact = frame.loc[frame["is_exact"].astype(bool)]
        censored = frame.loc[~frame["is_exact"].astype(bool)]
        observed_min, observed_max = finite_range(frame["p_activity"])
        exact_min, exact_max = finite_range(exact["p_activity"])
        censored_min, censored_max = finite_range(censored["p_activity"])
        rows.append(
            {
                "endpoint": endpoint,
                "rows": len(frame),
                "observed_min": observed_min,
                "observed_max": observed_max,
                "exact_rows": len(exact),
                "exact_min": exact_min,
                "exact_max": exact_max,
                "censored_rows": len(censored),
                "censored_threshold_min": censored_min,
                "censored_threshold_max": censored_max,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.6f")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
