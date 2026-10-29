#!/usr/bin/env python3
"""Generate reviewer-requested affine-frontier and threshold-audit figures."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "revision_source_data"
QA = ROOT / "qa"
QA.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "qa_tools"))

from audit_panel_alignment import require_matplotlib_panel_alignment


BLUE = "#2166D1"
CYAN = "#24BFC4"
ORANGE = "#F2A541"
CORAL = "#EF6A75"
GRAY = "#7D8793"
INK = "#1B2A41"


def set_style() -> None:
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
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def finish_axes(axes: list[plt.Axes]) -> None:
    for index, ax in enumerate(axes):
        ax.tick_params(direction="out", colors=INK)
        for spine in ax.spines.values():
            spine.set_color(INK)
            spine.set_linewidth(0.55)
        ax.text(
            -0.13,
            1.035,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=INK,
        )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=QA / f"{stem}_alignment.json",
        overlay_svg=QA / f"{stem}_alignment.svg",
        require_panel_labels=True,
        strict=False,
    )
    fig.savefig(ROOT / f"{stem}.pdf", bbox_inches="tight", pad_inches=5 / 72)
    fig.savefig(
        ROOT / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=5 / 72,
    )
    plt.close(fig)


def plot_affine_frontiers() -> None:
    affine = pd.read_csv(DATA / "r2_1_affine_target_balanced_summary.csv")
    methods = pd.read_csv(DATA / "r2_8_r2_10d_target_balanced_summary.csv").set_index(
        "method"
    )
    matched = methods.loc["Censored-as-exact LightGBM"]
    canal = methods.loc["CANAL"]

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.68))
    specifications = [
        (
            affine.loc[affine["strategy"].eq("maximize_cdir")].sort_values(
                "mae_tolerance"
            ),
            "test_cdir_mean",
            "cDir",
            (matched["exact_mae_mean"], matched["cdir_mean"]),
            (canal["exact_mae_mean"], canal["cdir_mean"]),
        ),
        (
            affine.loc[affine["strategy"].eq("minimize_violation")].sort_values(
                "mae_tolerance"
            ),
            "test_violation_distance_mean",
            "Boundary-violation distance",
            (
                matched["exact_mae_mean"],
                matched["violation_distance_mean"],
            ),
            (canal["exact_mae_mean"], canal["violation_distance_mean"]),
        ),
    ]
    for ax, (curve, y_column, y_label, matched_point, canal_point) in zip(
        axes, specifications
    ):
        ax.plot(
            curve["test_exact_mae_mean"],
            curve[y_column],
            color=ORANGE,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=ORANGE,
            markeredgewidth=0.55,
            zorder=2,
        )
        ax.scatter(
            *matched_point,
            color=GRAY,
            edgecolor=INK,
            linewidth=0.35,
            s=12,
            zorder=3,
        )
        ax.scatter(
            *canal_point,
            color=BLUE,
            edgecolor=INK,
            linewidth=0.35,
            s=14,
            zorder=4,
        )
        ax.set_xlabel("Exact-record MAE")
        ax.set_ylabel(y_label)
        ax.grid(False)

    handles = [
        Line2D(
            [0],
            [0],
            color=ORANGE,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=ORANGE,
            markeredgewidth=0.55,
            label="Affine, validation-selected",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            color=GRAY,
            markeredgecolor=INK,
            markeredgewidth=0.35,
            label="Matched",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            color=BLUE,
            markeredgecolor=INK,
            markeredgewidth=0.35,
            label="CANAL",
        ),
    ]
    finish_axes(list(axes))
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=3,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.20, top=0.80, wspace=0.26)
    save_figure(fig, "figS09_affine_frontiers")


def plot_threshold_audit() -> None:
    frequency = pd.read_csv(DATA / "r2_2_threshold_frequency.csv")
    frequency = frequency.loc[
        frequency["standard_units"].eq("nM")
        & frequency["standard_value"].gt(0)
    ].copy()
    frequency["log10_nm"] = np.log10(frequency["standard_value"])
    comparison = pd.read_csv(DATA / "r2_2_threshold_vs_explicit_max_rows.csv")
    cross_tab = (
        comparison.groupby(
            ["standard_value", "explicit_max_nM", "threshold_equals_explicit_max"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "rows"})
    )

    bin_width = 0.25
    bins = np.arange(
        np.floor(frequency["log10_nm"].min() / bin_width) * bin_width,
        np.ceil(frequency["log10_nm"].max() / bin_width) * bin_width + bin_width,
        bin_width,
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.68))
    ax = axes[0]
    for relation, color, label in [
        ("<", CYAN, "Left-censored pActivity"),
        (">", CORAL, "Right-censored pActivity"),
    ]:
        subset = frequency.loc[frequency["p_relation"].eq(relation)]
        ax.hist(
            subset["log10_nm"],
            bins=bins,
            weights=subset["rows"],
            histtype="step",
            linewidth=0.75,
            color=color,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"Threshold concentration, $\log_{10}$(nM)")
    ax.set_ylabel("Records")
    ax.legend(loc="upper left", frameon=False, handlelength=1.4)

    ax = axes[1]
    limits = [5000.0, 50000.0]
    ax.plot(
        limits,
        limits,
        color=GRAY,
        linestyle="--",
        linewidth=0.55,
        label="Identity",
        zorder=1,
    )
    for _, row in cross_tab.iterrows():
        match = bool(row["threshold_equals_explicit_max"])
        color = BLUE if match else CORAL
        marker = "o" if match else "s"
        size = 18 + 2 * np.sqrt(row["rows"])
        ax.scatter(
            row["standard_value"],
            row["explicit_max_nM"],
            s=size,
            color=color,
            marker=marker,
            edgecolor=INK,
            linewidth=0.35,
            zorder=3,
        )
        if match:
            offset = (5, -12)
        elif row["explicit_max_nM"] > 20000:
            offset = (5, -9)
        else:
            offset = (5, -10)
        ax.annotate(
            f"n={int(row['rows'])}",
            (row["standard_value"], row["explicit_max_nM"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
            color=INK,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Reported threshold (nM)")
    ax.set_ylabel("Explicit assay maximum (nM)")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color=BLUE,
                markeredgecolor=INK,
                markeredgewidth=0.35,
                label="Match",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                color=CORAL,
                markeredgecolor=INK,
                markeredgewidth=0.35,
                label="Mismatch",
            ),
            Line2D(
                [0],
                [0],
                color=GRAY,
                linestyle="--",
                linewidth=0.55,
                label="Identity",
            ),
        ],
        loc="upper left",
        frameon=False,
        handlelength=1.2,
    )

    finish_axes(list(axes))
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.20, top=0.95, wspace=0.27)
    save_figure(fig, "figS10_threshold_audit")


def main() -> None:
    set_style()
    plot_affine_frontiers()
    plot_threshold_audit()
    print(ROOT)


if __name__ == "__main__":
    main()
