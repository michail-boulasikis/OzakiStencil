#!/usr/bin/env python3
"""
Plot performance bar chart from performance.sh results.tsv.

Supports one or more result files plotted side by side as subplots. Each
subplot is stacked vertically in black and white when --thesis is enabled. Each
subplot compares the 7×7 CuTe coefficient-kernel variants. Direct baselines
are measured with the 7x7LmNone stencil, then appear at the matching precision
level:
  - s=5  (≈FP32):  + Direct FP32
  - s=9  (≈FP64):  + Direct FP64; 7×7 also adds ConvStencil and EBISU
  - s=17 (≈FP128): + Direct FP128 when supported by the device

Usage:
    python scripts/plot_perf_bar.py R1.tsv [-o perf_bar.pdf]
    python scripts/plot_perf_bar.py R1.tsv R2.tsv --gpu A100,H200 [-o perf_bar.pdf]
    python scripts/plot_perf_bar.py R1.tsv R2.tsv --gpu A100,H200 --thesis -o thesis_perf.pdf
"""

import argparse
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks", font_scale=1.2)
sns.set_context("paper")
plt.rcParams["ps.fonttype"] = 42

OZAKI_VARIANTS = (
    ("7x7Lm1_0", "Ozaki DSO1"),
    ("7x7Lm2_0", "Ozaki DSO2"),
    ("7x7LmNone_0", "Ozaki"),
)
BASELINE_STENCIL = "7x7LmNone_0"

COLORS = {
    "Ozaki DSO1": "#9eb9f3",
    "Ozaki DSO2": "#87c55f",
    "Ozaki": "#dcb0f2",
    "Direct FP32": "#f89c74",
    "Direct FP64": "#f89c74",
    "Direct FP128": "#f89c74",
    "ConvStencil": "#f6cf71",
    "EBISU": "#66c5cc",
}

BAR_WIDTH = 0.40
BAR_GAP = 0.0
GROUP_GAP = 0.55

HATCHES = {
    "Ozaki DSO1": "",
    "Ozaki DSO2": "oo",
    "Ozaki": "//",
    "Direct FP32": "xx",
    "Direct FP64": "xx",
    "Direct FP128": "xx",
    "ConvStencil": "..",
    "EBISU": "\\\\",
}

PRECISION_LABELS = {
    5: r"($\approx$FP32)",
    9: r"($\approx$FP64)",
    17: r"($\approx$FP128)",
}


def lookup(df, framework, stencil, detail="-"):
    mask = (
        (df["Framework"] == framework)
        & (df["Stencil"] == stencil)
        & (df["Detail"] == detail)
    )
    vals = df.loc[mask, "GStPt/s"]
    if not len(vals) or pd.isna(vals.iloc[0]):
        return None
    return float(vals.iloc[0])


def build_groups(df, slice_counts):
    groups = []
    for s in slice_counts:
        bars = [
            (display, lookup(df, "OzakiStencil", stencil, f"s={s}"))
            for stencil, display in OZAKI_VARIANTS
        ]

        # Direct kernels do not depend on the Ozaki coefficient layout, so
        # performance.sh records them only for the LmNone stencil.
        if s == 5:
            bars.append(("Direct FP32", lookup(df, "Direct FP32", BASELINE_STENCIL)))
        if s == 9:
            bars.append(("Direct FP64", lookup(df, "Direct FP64", BASELINE_STENCIL)))
            bars.append(("ConvStencil", lookup(df, "ConvStencil", "box2d3r")))
            bars.append(("EBISU", lookup(df, "EBISU", "box2d3r")))
        if s == 17:
            fp128 = lookup(df, "Direct FP128", BASELINE_STENCIL)
            if fp128 is not None:
                bars.append(("Direct FP128", fp128))

        label = f"s={s}"
        if s in PRECISION_LABELS:
            label += f"\n{PRECISION_LABELS[s]}"
        groups.append((label, bars))
    return groups


def plot_groups(ax, groups, thesis=False):
    """Draw variable-width grouped bars. Returns {label: patch} for legend."""
    seen = {}
    x = 0.0

    for group_label, bars in groups:
        positions = []
        for i, (name, val) in enumerate(bars):
            pos = x + i * (BAR_WIDTH + BAR_GAP)
            positions.append(pos)
            if val is None:
                continue
            color = COLORS.get(name, "#999999")
            hatch = HATCHES.get(name, "")
            rect = ax.bar(
                pos,
                val,
                BAR_WIDTH,
                color=color,
                hatch=hatch,
                edgecolor="black" if thesis else "#555555",
                linewidth=0.5,
            )
            ax.annotate(
                f"{val:.1f}",
                (pos, val),
                xytext=(0, 3 if thesis else 0),
                textcoords="offset points",
                ha="center" if thesis else "left",
                va="bottom",
                fontsize=6,
                fontweight="bold",
                rotation=90 if thesis else 45,
            )
            if name not in seen:
                seen[name] = rect[0]

        center = (positions[0] + positions[-1]) / 2
        ax.annotate(
            group_label,
            (center, 0),
            xytext=(0, -20),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=11,
        )

        x = positions[-1] + BAR_WIDTH + GROUP_GAP

    ax.set_xlim(-BAR_WIDTH, x - GROUP_GAP + BAR_WIDTH)
    ax.set_xticks([])
    return seen


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "results_tsv", nargs="+", help="Path(s) to results.tsv from performance.sh"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="perf_bar.pdf",
        help="Output file (default: perf_bar.pdf)",
    )
    parser.add_argument(
        "--thesis",
        action="store_true",
        help="Use black-and-white hatched bars with one subplot per row",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        default="RTX 5080",
        help="Comma-separated GPU names, one per input file (default: RTX 5080)",
    )
    args = parser.parse_args()

    gpu_names = [g.strip() for g in args.gpu.split(",")]
    n = len(args.results_tsv)
    # Pad gpu_names if fewer than files, or truncate if more
    if len(gpu_names) < n:
        gpu_names.extend([gpu_names[-1]] * (n - len(gpu_names)))
    gpu_names = gpu_names[:n]

    if args.thesis:
        plt.rcParams.update({
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "hatch.color": "black",
        })
        fig, axes = plt.subplots(n, 1, figsize=(7, 3.2 * n), squeeze=False)
    else:
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 3.2), squeeze=False)
    axes = axes.ravel()

    all_handles = {}
    for idx, (tsv_path, gpu) in enumerate(zip(args.results_tsv, gpu_names)):
        ax = axes[idx]
        df = pd.read_csv(tsv_path, sep="\t")
        df["GStPt/s"] = pd.to_numeric(df["GStPt/s"], errors="coerce")

        slice_counts = sorted(
            {
                int(match.group(1))
                for detail in df.loc[
                    df["Framework"] == "OzakiStencil", "Detail"
                ].unique()
                if (match := re.fullmatch(r"s=(\d+)", str(detail))) is not None
            }
        )

        groups = build_groups(df, slice_counts)
        handles = plot_groups(ax, groups, thesis=args.thesis)
        all_handles.update(handles)

        ax.set_xlabel("Data slices")
        if idx == 0 or args.thesis:
            ax.set_ylabel("GPt/s")
        ax.set_title(f"{gpu}", pad=16)
        sns.despine(ax=ax)

    # Unify y-axis limits across all subplots
    # global_ymax = max(ax.get_ylim()[1] for ax in axes)
    # for ax in axes:
    #    ax.set_ylim(0, global_ymax)

    # Single shared legend from the last subplot
    order = [display for _, display in OZAKI_VARIANTS] + [
        "Direct FP32",
        "Direct FP64",
        "Direct FP128",
        "ConvStencil",
        "EBISU",
    ]
    legend_handles = [all_handles[k] for k in order if k in all_handles]
    legend_labels = [k for k in order if k in all_handles]
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(right=0.82)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.82, 0.5),
        fontsize=9,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.2,
    )

    # fig.suptitle("OzakiStencil Performance Comparisons", fontweight="bold", y=0.90)
    plt.show()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
