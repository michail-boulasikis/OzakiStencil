#!/usr/bin/env python3
"""Plot aggregate correctness-error box plots over all profiled stencils."""

import argparse
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(
    style="ticks",
    context="paper",
    font_scale=1.0,
    rc={
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.65,
    },
)
plt.rcParams["ps.fonttype"] = 42  # embed TrueType fonts so text is selectable in LaTeX

# Colors from the pastel-8 palette, one per T value. Keep this mapping stable
# so the base and --longer-loops plots use the same colors.
T_COLORS = {
    100000: "#9eb9f3",
    10000: "#87c55f",
    1000: "#66c5cc",
    100: "#f6cf71",
    10: "#f89c74",
    1: "#dcb0f2",
}

DEFAULT_REFS_CSV = "data/results/precision_refs.csv"
REQUIRED_COLUMNS = {"stencil", "T", "slices", "rmse", "max_abs_diff"}
METRICS = (
    ("rmse", "Error", "Normalized RMS Error"),
    ("max_abs_diff", "Error", "Worst Relative Pointwise Error"),
)
MACHINE_EPSILONS = (
    ("FP32", np.finfo(np.float32).eps, ":"),
    ("FP64", np.finfo(np.float64).eps, "--"),
    ("FP80", np.finfo(np.longdouble).eps, "-."),
)

# Sized for a conventional single-column research-paper figure.
PAPER_COLUMN_WIDTH = 4.25
PAPER_FIGURE_HEIGHT = 4.25
MAX_X_TICK_LABELS = 8
PLOT_SLICE_VALUES = tuple(range(5, 18, 2))
ZOOM_SLICE = 17
ZOOM_INSET_BOUNDS = (0.55, 0.7, 0.40, 0.29)
BOX_EDGE_COLOR = "#404040"
ZOOM_EDGE_COLOR = "#b3b3b3"
ZOOM_FILL_COLOR = "#fff7df"
ZOOM_FILL_ALPHA = 0.28
ZOOM_Y_LIMITS = (1e-36, 1e-29)
Y_AXIS_BOTTOM = 1e-37
Y_AXIS_TOP = 1e1
REFERENCE_LINE_COLOR = "#888888"
REFERENCE_RATE_LINESTYLE = (1, (0.5, 0.5))
REFERENCE_RATE_OFFSET = 24000.0 * 128


def load_results(path: str) -> pd.DataFrame:
    """Load and validate finite, strictly positive errors for logarithmic plots."""
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()

    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            "Input CSV is missing required column(s): " + ", ".join(sorted(missing))
        )

    for column in ("T", "slices", "rmse", "max_abs_diff"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    numeric_columns = ["T", "slices", "rmse", "max_abs_diff"]
    invalid = data["stencil"].isna() | ~np.isfinite(data[numeric_columns]).all(axis=1)
    nonpositive = (data["rmse"] <= 0) | (data["max_abs_diff"] <= 0)
    dropped = int((invalid | nonpositive).sum())
    if dropped:
        warnings.warn(
            f"Dropping {dropped} row(s) with missing, non-numeric, or non-positive "
            "values that cannot be shown on a logarithmic axis.",
            stacklevel=2,
        )
        data = data.loc[~(invalid | nonpositive)].copy()

    if data.empty:
        raise ValueError("Input CSV has no positive finite error measurements to plot")

    data["T"] = data["T"].astype(int)
    data["slices"] = data["slices"].astype(int)
    return data


def add_slice_zoom(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    slice_positions: dict[int, int],
    timestep_values: list[int],
    palette: dict[int, str],
) -> None:
    """Highlight the selected slice and enlarge its distributions in an inset."""
    if ZOOM_SLICE not in slice_positions:
        return

    position = slice_positions[ZOOM_SLICE]
    zoom_y_low, zoom_y_high = ZOOM_Y_LIMITS
    ax.add_patch(
        mpatches.Rectangle(
            (position - 0.48, zoom_y_low),
            0.96,
            zoom_y_high - zoom_y_low,
            transform=ax.transData,
            facecolor=ZOOM_FILL_COLOR,
            edgecolor="none",
            alpha=ZOOM_FILL_ALPHA,
            zorder=0.5,
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (position - 0.48, zoom_y_low),
            0.96,
            zoom_y_high - zoom_y_low,
            transform=ax.transData,
            fill=False,
            edgecolor=ZOOM_EDGE_COLOR,
            linewidth=0.6,
            zorder=6,
        )
    )

    inset = ax.inset_axes(ZOOM_INSET_BOUNDS, zorder=7)
    zoom_data = data[data["slices"] == ZOOM_SLICE]
    sns.boxplot(
        data=zoom_data,
        x="T",
        y=metric,
        hue="T",
        order=timestep_values,
        hue_order=timestep_values,
        palette=palette,
        dodge=False,
        width=0.7,
        linewidth=0.55,
        saturation=1,
        boxprops={"edgecolor": BOX_EDGE_COLOR},
        medianprops={"color": BOX_EDGE_COLOR, "linewidth": 0.8},
        whiskerprops={"color": BOX_EDGE_COLOR, "linewidth": 0.5},
        capprops={"color": BOX_EDGE_COLOR, "linewidth": 0.5},
        flierprops={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": BOX_EDGE_COLOR,
            "markersize": 1.8,
            "markeredgewidth": 0.35,
        },
        log_scale=(False, True),
        legend=False,
        ax=inset,
    )
    inset.set_title(r"$s=17$ zoom", fontsize=5.5, pad=1.5)
    inset.set_xlabel(r"$T$", fontsize=5, labelpad=0.5)
    inset.set_ylabel("")
    inset.set_xticks(
        range(len(timestep_values)),
        [rf"$10^{{{int(np.log10(timestep))}}}$" for timestep in timestep_values],
    )
    inset.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
    inset.minorticks_off()
    inset.tick_params(axis="x", labelsize=4.2, length=1.8, width=0.45, pad=1)
    inset.tick_params(axis="y", labelsize=4.2, length=1.8, width=0.45, pad=1)
    inset.grid(
        axis="y",
        which="major",
        color="#dddddd",
        linestyle=(0, (1, 2)),
        linewidth=0.35,
    )
    inset.set_facecolor(ZOOM_FILL_COLOR)
    inset.patch.set_alpha(ZOOM_FILL_ALPHA)
    for spine in inset.spines.values():
        spine.set_color(ZOOM_EDGE_COLOR)
        spine.set_linewidth(0.55)

    connector = mpatches.ConnectionPatch(
        xyA=(0.5, 0),
        coordsA=inset.transAxes,
        xyB=(position, 1e-29),
        coordsB=ax.transData,
        color=ZOOM_EDGE_COLOR,
        linewidth=0.4,
        zorder=6,
    )
    connector.set_clip_on(False)
    ax.add_artist(connector)


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
        help="Output file containing both aggregate plots "
        "(default: rmse_vs_slices.eps)",
    )
    parser.add_argument(
        "--refs",
        default=DEFAULT_REFS_CSV,
        help="Accepted for compatibility with the correctness pipeline; aggregate "
        "box plots use the main results CSV only",
    )
    parser.add_argument(
        "--no-trend-lines",
        action="store_true",
        help="Suppress timestep-average trend lines over the box plots",
    )
    args = parser.parse_args()

    data = load_results(args.csv)
    data = data[data["slices"].isin(PLOT_SLICE_VALUES)].copy()
    if data.empty:
        raise ValueError(
            "Input CSV has no measurements at the selected slice counts: "
            + ", ".join(str(value) for value in PLOT_SLICE_VALUES)
        )
    slice_values = sorted(data["slices"].unique())
    if slice_values[0] <= 0:
        raise ValueError("Slice counts must be positive to plot the reference rate")
    timestep_values = sorted(data["T"].unique())
    unknown_timesteps = [t for t in timestep_values if t not in T_COLORS]
    if unknown_timesteps:
        raise ValueError(
            "No palette color is defined for timestep count(s): "
            + ", ".join(str(t) for t in unknown_timesteps)
        )

    palette = {t: T_COLORS[t] for t in timestep_values}
    slice_positions = {slice_value: position for position, slice_value in enumerate(slice_values)}
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(PAPER_COLUMN_WIDTH, PAPER_FIGURE_HEIGHT),
        sharex=True,
    )

    for ax, (metric, label, title) in zip(axes, METRICS):
        sns.boxplot(
            data=data,
            x="slices",
            y=metric,
            hue="T",
            order=slice_values,
            hue_order=timestep_values,
            palette=palette,
            width=0.92,
            gap=0.10,
            linewidth=0.45,
            saturation=1,
            boxprops={"edgecolor": BOX_EDGE_COLOR},
            medianprops={"color": BOX_EDGE_COLOR, "linewidth": 0.75},
            whiskerprops={"color": BOX_EDGE_COLOR, "linewidth": 0.45},
            capprops={"color": BOX_EDGE_COLOR, "linewidth": 0.45},
            flierprops={
                "marker": "o",
                "markerfacecolor": "none",
                "markeredgecolor": BOX_EDGE_COLOR,
                "markersize": 2.2,
                "markeredgewidth": 0.4,
            },
            log_scale=(False, True),
            legend=False,
            ax=ax,
        )

        if not args.no_trend_lines:
            # Overlay the mean error at each slice count for every timestep.
            # The boxes show the stencil-wise distribution; these lines show
            # its central trend using the same timestep colors.
            means = (
                data.groupby(["T", "slices", "stencil"], observed=True)[metric]
                .mean()
                .reset_index(name=metric)
                .groupby(["T", "slices"], as_index=False, observed=True)
                [metric]
                .mean()
            )
            for timestep in timestep_values:
                timestep_means = means[means["T"] == timestep].sort_values("slices")
                ax.plot(
                    timestep_means["slices"].map(slice_positions),
                    timestep_means[metric],
                    color=palette[timestep],
                    marker="o",
                    markersize=3.2,
                    linewidth=0.65,
                    alpha=0.9,
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                    zorder=1,
                )

        # Scale the rate guide from the median error at the smallest slice
        # count, then offset it upward so it remains distinct from the data.
        # Each additional slice should reduce the error by 128 = 2^7.
        reference_anchor = data.loc[
            data["slices"] == slice_values[0], metric
        ].median()
        slice_offsets = np.asarray(slice_values, dtype=float) - slice_values[0]
        reference_rate = (
            REFERENCE_RATE_OFFSET
            * reference_anchor
            * np.exp2(-7 * slice_offsets)
        )
        ax.plot(
            range(len(slice_values)),
            reference_rate,
            color=REFERENCE_LINE_COLOR,
            linestyle=REFERENCE_RATE_LINESTYLE,
            linewidth=1.0,
            zorder=4,
        )

        ax.set_title(title, loc="left", fontweight="bold", pad=3)
        ax.set_ylabel(label)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
        # With errors spanning tens of decades, the nine minor log ticks per
        # decade merge into a distracting solid bar at the axis spine.
        ax.minorticks_off()
        ax.set_axisbelow(True)
        ax.grid(
            axis="y",
            which="major",
            color="#d9d9d9",
            linestyle=(0, (1, 2)),
            linewidth=0.45,
        )
        ax.tick_params(axis="both", which="major", length=3, width=0.65, pad=2)
        for _, epsilon, linestyle in MACHINE_EPSILONS:
            ax.axhline(
                epsilon,
                color=REFERENCE_LINE_COLOR,
                linestyle=linestyle,
                linewidth=1.0,
                zorder=4,
            )
        ax.set_ylim(bottom=Y_AXIS_BOTTOM, top=Y_AXIS_TOP)
        sns.despine(ax=ax)
        add_slice_zoom(
            ax,
            data,
            metric,
            slice_positions,
            timestep_values,
            palette,
        )

    axes[0].tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("Slices")
    tick_stride = max(1, int(np.ceil(len(slice_values) / MAX_X_TICK_LABELS)))
    tick_indices = list(range(0, len(slice_values), tick_stride))
    if tick_indices[-1] != len(slice_values) - 1:
        if len(slice_values) - 1 - tick_indices[-1] < tick_stride:
            tick_indices.pop()
        tick_indices.append(len(slice_values) - 1)
    axes[-1].set_xticks(
        tick_indices,
        [str(slice_values[index]) for index in tick_indices],
    )
    for tick_label in axes[-1].get_xticklabels():
        if tick_label.get_text() == str(ZOOM_SLICE):
            tick_label.set_color(ZOOM_EDGE_COLOR)
            tick_label.set_fontweight("bold")

    legend_handles = [
        mpatches.Patch(facecolor=T_COLORS[t], edgecolor="none", label=rf"$T={t}$")
        for t in timestep_values
    ]
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            color=REFERENCE_LINE_COLOR,
            linestyle=linestyle,
            linewidth=1.0,
            label=label,
        )
        for label, _, linestyle in MACHINE_EPSILONS
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=REFERENCE_LINE_COLOR,
            linestyle=REFERENCE_RATE_LINESTYLE,
            linewidth=1.0,
            label=r"$2^{-7s}$",
        )
    )
    legend_columns = 3
    legend_rows = int(np.ceil(len(legend_handles) / legend_columns))
    # Matplotlib fills multi-column legends by column. Reorder the handles so
    # the displayed legend still reads left-to-right, then top-to-bottom.
    legend_handles = [
        legend_handles[row * legend_columns + column]
        for column in range(legend_columns)
        for row in range(legend_rows)
        if row * legend_columns + column < len(legend_handles)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=legend_columns,
        fontsize=6.5,
        frameon=False,
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0,
    )

    fig.subplots_adjust(left=0.22, right=0.98, top=0.96, bottom=0.18, hspace=0.22)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(
        f"Saved aggregate distributions for {data['stencil'].nunique()} stencil(s) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
