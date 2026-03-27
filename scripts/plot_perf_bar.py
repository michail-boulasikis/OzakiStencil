#!/usr/bin/env python3
"""
Plot performance bar chart from eval.sh results.tsv.

Supports one or more result files plotted side by side as subplots.
Each subplot shows bars grouped by slice count (s=5,7,10,13,17).
Each group shows Ozaki, Ozaki+DSO1, and Ozaki+DSO2 (discrete subgroup optimization).
Baselines appear at the matching precision level:
  - s=5  (≈FP32):  + Direct FP32
  - s=10 (≈FP64):  + Direct FP64, ConvStencil (box2d3r), EBISU (box2d3r)

Usage:
    python scripts/plot_perf_bar.py R1.tsv [-o perf_bar.pdf]
    python scripts/plot_perf_bar.py R1.tsv R2.tsv --gpu A100,H200 [-o perf_bar.pdf]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks", font_scale=1.2)
sns.set_context("paper")

OZAKI_VARIANTS = {
    "7x7Lm1_0": "Ozaki + DSO1",
    "7x7Lm2_0": "Ozaki + DSO2",
    "7x7LmNone_0": "Ozaki",
}

COLORS = {
    "Ozaki + DSO2": "#87c55f",
    "Ozaki + DSO1": "#9eb9f3",
    "Ozaki": "#dcb0f2",
    "Direct FP32": "#f89c74",
    "Direct FP64": "#f89c74",
    "ConvStencil": "#f6cf71",
    "EBISU": "#66c5cc",
}

BAR_WIDTH = 0.40
BAR_GAP = 0.0
GROUP_GAP = 0.55

HATCHES = {
    "Ozaki + DSO2": "oo",
    "Ozaki + DSO1": "",
    "Ozaki": "//",
    "Direct FP32": "xx",
    "Direct FP64": "xx",
    "ConvStencil": "..",
    "EBISU": "\\\\",
}

PRECISION_LABELS = {
    5: r"($\approx$FP32)",
    10: r"($\approx$FP64)",
    17: r"($\approx$FP128)",
}


def lookup(df, framework, stencil, detail="-"):
    mask = (
        (df["Framework"] == framework)
        & (df["Stencil"] == stencil)
        & (df["Detail"] == detail)
    )
    vals = df.loc[mask, "GStPt/s"]
    return float(vals.iloc[0]) if len(vals) else None


def build_groups(df, slice_counts):
    groups = []
    for s in slice_counts:
        bars = []
        # Ozaki variants (DSO first, then plain)
        for tsv_name, display in OZAKI_VARIANTS.items():
            bars.append((display, lookup(df, "OzakiConvStencil", tsv_name, f"s={s}")))

        # Baselines at matching precision
        if s == 5:
            # Use any ozaki stencil for the Direct FP32 baseline (same perf)
            fp32 = lookup(df, "Direct FP32", next(iter(OZAKI_VARIANTS)))
            bars.append(("Direct FP32", fp32))
        if s == 10:
            fp64 = lookup(df, "Direct FP64", next(iter(OZAKI_VARIANTS)))
            bars.append(("Direct FP64", fp64))
            bars.append(("ConvStencil", lookup(df, "ConvStencil", "box2d3r")))
            bars.append(("EBISU", lookup(df, "EBISU", "box2d3r")))

        label = f"s={s}"
        if s in PRECISION_LABELS:
            label += f"\n{PRECISION_LABELS[s]}"
        groups.append((label, bars))
    return groups


def plot_groups(ax, groups):
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
                edgecolor="#555555",
                linewidth=0.5,
            )
            ax.text(
                pos,
                val,
                f"{val:.1f}",
                ha="left",
                va="bottom",
                fontsize=6,
                fontweight="bold",
                rotation=30,
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
        "results_tsv", nargs="+", help="Path(s) to results.tsv from eval.sh"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="perf_bar.pdf",
        help="Output file (default: perf_bar.pdf)",
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

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 3.2), squeeze=False)
    axes = axes[0]  # flatten from (1, n) to (n,)

    all_handles = {}
    for idx, (tsv_path, gpu) in enumerate(zip(args.results_tsv, gpu_names)):
        ax = axes[idx]
        df = pd.read_csv(tsv_path, sep="\t")
        df["GStPt/s"] = pd.to_numeric(df["GStPt/s"], errors="coerce")

        slice_counts = sorted(
            int(d.split("=")[1])
            for d in df.loc[df["Framework"] == "OzakiConvStencil", "Detail"].unique()
        )

        groups = build_groups(df, slice_counts)
        handles = plot_groups(ax, groups)
        all_handles.update(handles)

        ax.set_xlabel("Data slices")
        if idx == 0:
            ax.set_ylabel("GStPt/s")
        ax.set_title(f"{gpu}", pad=16)
        sns.despine(ax=ax)

    # Single shared legend from the last subplot
    order = list(OZAKI_VARIANTS.values()) + [
        "Direct FP32",
        "Direct FP64",
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
        handletextpad=0.4,
    )

    # fig.suptitle("OzakiStencil Performance Comparisons", fontweight="bold", y=0.90)
    plt.show()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
