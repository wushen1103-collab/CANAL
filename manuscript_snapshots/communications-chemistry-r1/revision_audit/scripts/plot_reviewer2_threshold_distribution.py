#!/usr/bin/env python3
"""Plot left- and right-censored IC50 threshold distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    audit_dir = args.audit_dir.resolve()
    output_dir = audit_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    frequency = pd.read_csv(audit_dir / "r2_2_threshold_frequency.csv")
    frequency = frequency.loc[
        frequency["standard_units"].eq("nM") & frequency["standard_value"].gt(0)
    ].copy()
    frequency["log10_nm"] = np.log10(frequency["standard_value"])
    bins = np.arange(
        np.floor(frequency["log10_nm"].min() * 4) / 4,
        np.ceil(frequency["log10_nm"].max() * 4) / 4 + 0.26,
        0.25,
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.65), sharex=True)
    specifications = [
        ("<", "#24BFC4", "Left-censored pActivity"),
        (">", "#EF6A75", "Right-censored pActivity"),
    ]
    for index, (axis, (relation, color, label)) in enumerate(zip(axes, specifications)):
        subset = frequency.loc[frequency["p_relation"].eq(relation)]
        counts, edges = np.histogram(
            subset["log10_nm"], bins=bins, weights=subset["rows"]
        )
        axis.bar(
            edges[:-1], counts, width=np.diff(edges), align="edge",
            color=color, edgecolor="#1B2A41", linewidth=0.35,
            label=label,
        )
        axis.text(0.02, 0.96, f"({chr(97 + index)})", transform=axis.transAxes,
                  ha="left", va="top", fontweight="bold")
        axis.legend(loc="upper left", bbox_to_anchor=(0.08, 1.0), frameon=False)
        axis.set_xlabel(r"Threshold concentration, $\log_{10}$(nM)")
        axis.tick_params(direction="out", length=2.5)
        for spine in axis.spines.values():
            spine.set_linewidth(0.55)
    axes[0].set_ylabel("Records")
    axes[1].set_ylabel("Records")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.98, wspace=0.23)
    output = output_dir / "r2_2_threshold_distribution"
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=5 / 72)
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=5 / 72)
    plt.close(fig)
    print(output_dir)


if __name__ == "__main__":
    main()
