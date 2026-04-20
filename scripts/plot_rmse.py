#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="ticks", font_scale=1.5)
sns.set_context("talk")
plt.rcParams["ps.fonttype"] = 42  # embed TrueType fonts so text is selectable in LaTeX

# ── Configurable: rename stencil keys to display titles ──────────────────────
STENCIL_TITLES = {
    "heat": "Heat Equation",
    "ad": "Advection-Diffusion",
    "7x7Lm1_0": r"7×7 Box (DSO1)",
    "3x3x3LmNone_0": r"3×3×3 Box",
}

# Desired subplot order (keys absent in the CSV are skipped)
STENCIL_ORDER = ["heat", "ad", "7x7Lm1_0", "3x3x3Lm2_0"]

# Default path for precision reference data
DEFAULT_REFS_CSV = "data/precision_refs.csv"

# Colors from the plot_perf_bar.py palette, one per T value
T_COLORS = {
    1000: "#66c5cc",  # blue   (≈ "Ozaki + DSO1")
    100: "#f6cf71",  # green  (≈ "Ozaki + DSO2")
    10: "#f89c74",  # orange (≈ "Direct FP32/64")
    1: "#dcb0f2",  # purple (≈ "Ozaki")
}

# Markers for colorblind accessibility
T_MARKERS = {
    1: "o",  # circle
    10: "s",  # square
    100: "^",  # triangle-up
    1000: "D",  # diamond
}

# Linestyles for precision reference lines (per precision type)
PREC_LINESTYLES = {
    "float": (0, (4, 2)),  # dashed
    "double": (0, (2, 1, 1, 1)),  # dash-dot
    "longdouble": (0, (1, 1)),  # dotted
}

PREC_LABELS = {
    "float": "FP32",
    "double": "FP64",
    "longdouble": "FP80",
}

MARKER_SIZE = 6.0
LINE_WIDTH = 2.2
REF_WIDTH = 1.6
REF_ALPHA = 0.65
SHARED_Y_AXIS = False  # set True to share the y-axis across all subplots


def load_refs(path: str) -> dict:
    """Return nested dict: refs[stencil][T][precision] = rmse."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    refs: dict = {}
    for _, row in df.iterrows():
        s, t, p, v = row["stencil"], int(row["T"]), row["precision"], float(row["rmse"])
        refs.setdefault(s, {}).setdefault(t, {})[p] = v
    return refs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "csv", nargs="?", default="test.csv", help="Input CSV (default: test.csv)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="rmse_vs_slices.eps",
        help="Output file (default: rmse_vs_slices.eps)",
    )
    parser.add_argument(
        "--refs",
        default=DEFAULT_REFS_CSV,
        help=f"Precision reference CSV (default: {DEFAULT_REFS_CSV})",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()
    refs = load_refs(args.refs)

    # Order subplots; append any unexpected stencils at the end
    present = df["stencil"].unique()
    ordered = [s for s in STENCIL_ORDER if s in present]
    ordered += [s for s in present if s not in STENCIL_ORDER]
    n = len(ordered)

    sharey_arg = "row" if SHARED_Y_AXIS else False
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(3.8 * n, 3.4),
        sharey=sharey_arg,
        squeeze=False,
    )
    axes = axes[0]

    T_vals = sorted(df["T"].unique())

    for ax, stencil in zip(axes, ordered):
        sub = df[df["stencil"] == stencil]

        for t in T_vals:
            tdata = sub[sub["T"] == t].sort_values("slices")
            if tdata.empty:
                continue
            ax.plot(
                tdata["slices"],
                tdata["rmse"],
                color=T_COLORS.get(t, "#999999"),
                marker=T_MARKERS.get(t, "x"),
                markersize=MARKER_SIZE,
                markeredgewidth=0.5,
                markeredgecolor="white",
                linewidth=LINE_WIDTH,
                markevery=1,
                label=f"$T={t}$",
                zorder=3,
            )

        stencil_refs = refs.get(stencil, {})
        for prec in ("float", "double", "longdouble"):
            values = [
                stencil_refs[t][prec]
                for t in T_vals
                if t in stencil_refs
                and prec in stencil_refs[t]
                and stencil_refs[t][prec] > 0
            ]
            if not values:
                continue
            avg_rmse = sum(values) / len(values)
            ax.axhline(
                avg_rmse,
                color="#444444",
                linestyle=PREC_LINESTYLES[prec],
                linewidth=REF_WIDTH,
                alpha=REF_ALPHA,
                zorder=2,
            )

        ax.set_yscale("log")
        ax.set_xlabel("Slices", fontsize=13)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=7))
        ax.tick_params(axis="both", labelsize=11)
        title = STENCIL_TITLES.get(stencil, stencil)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=5)
        sns.despine(ax=ax)

    # Unify y-axis limits across all subplots
    all_ymins, all_ymaxs = [], []
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        all_ymins.append(ymin)
        all_ymaxs.append(ymax)
    global_ymin = min(all_ymins)
    global_ymax = max(all_ymaxs)
    for ax in axes:
        ax.set_ylim(global_ymin, global_ymax)

    axes[0].set_ylabel("RMSE", fontsize=13)

    # Section 1: T-value lines (colored, with markers)
    handles_T, labels_T = [], []
    seen: set = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles_T.append(h)
                labels_T.append(l)
    handles_T = sorted(
        zip(labels_T, handles_T), key=lambda x: int(x[0].strip("$T=").strip("$"))
    )
    labels_T = [x[0] for x in handles_T]
    handles_T = [x[1] for x in handles_T]

    # Section 2: precision reference lines (neutral dark gray, varying dash)
    handles_prec = [
        mlines.Line2D(
            [],
            [],
            color="#444444",
            linestyle=PREC_LINESTYLES[p],
            linewidth=REF_WIDTH + 0.3,
            label=PREC_LABELS[p],
        )
        for p in ("float", "double", "longdouble")
    ]
    labels_prec = [PREC_LABELS[p] for p in ("float", "double", "longdouble")]

    all_handles = handles_T + handles_prec
    all_labels = labels_T + labels_prec

    legend = fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        ncol=len(all_handles),
        fontsize=11,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.2,
    )

    # Measure the legend's actual height after drawing, then reserve exactly
    # enough space at the bottom so the legend sits flush below the subplots.
    fig.tight_layout()
    fig.canvas.draw()
    legend_top_frac = legend.get_window_extent().ymax / (
        fig.get_size_inches()[1] * fig.dpi
    )
    fig.subplots_adjust(bottom=legend_top_frac + 0.15)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
