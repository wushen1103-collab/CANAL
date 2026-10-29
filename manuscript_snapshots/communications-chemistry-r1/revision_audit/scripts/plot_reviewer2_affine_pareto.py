#!/usr/bin/env python3
"""Plot validation-selected affine trade-off curves for Reviewer 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


BLUE = "#2166D1"
ORANGE = "#F2A541"
GRAY = "#7D8793"
INK = "#1B2A41"


def style() -> None:
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
            "lines.linewidth": 0.75,
            "lines.markersize": 3.0,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def draw(
    curve: pd.DataFrame,
    y_column: str,
    y_label: str,
    matched: tuple[float, float],
    canal: tuple[float, float],
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.75))
    ax.plot(
        curve["test_exact_mae_mean"],
        curve[y_column],
        color=ORANGE,
        marker="o",
        markeredgecolor="white",
        markeredgewidth=0.35,
        label="Affine, validation-selected",
        zorder=2,
    )
    ax.scatter(*matched, color=GRAY, edgecolor="white", linewidth=0.4, s=20, label="Matched", zorder=3)
    ax.scatter(*canal, color=BLUE, edgecolor="white", linewidth=0.4, s=24, label="CANAL", zorder=4)
    ax.set_xlabel("Exact-record MAE")
    ax.set_ylabel(y_label)
    ax.legend(loc="best", frameon=False, handlelength=1.5, borderaxespad=0.4)
    ax.tick_params(direction="out", length=2.5, colors=INK)
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(0.55)
    ax.grid(False)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.18, top=0.98)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=5 / 72)
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=5 / 72)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    audit_dir = args.audit_dir.resolve()
    output_dir = audit_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    affine = pd.read_csv(audit_dir / "r2_1_affine_target_balanced_summary.csv")
    methods = pd.read_csv(audit_dir / "r2_8_r2_10d_target_balanced_summary.csv").set_index("method")
    matched = methods.loc["Censored-as-exact LightGBM"]
    canal = methods.loc["CANAL"]
    style()
    draw(
        affine.loc[affine["strategy"].eq("maximize_cdir")].sort_values("mae_tolerance"),
        "test_cdir_mean",
        "Censored-direction accuracy",
        (matched["exact_mae_mean"], matched["cdir_mean"]),
        (canal["exact_mae_mean"], canal["cdir_mean"]),
        output_dir / "r2_1_affine_mae_vs_cdir",
    )
    draw(
        affine.loc[affine["strategy"].eq("minimize_violation")].sort_values("mae_tolerance"),
        "test_violation_distance_mean",
        "Boundary-violation distance",
        (matched["exact_mae_mean"], matched["violation_distance_mean"]),
        (canal["exact_mae_mean"], canal["violation_distance_mean"]),
        output_dir / "r2_1_affine_mae_vs_violation",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
