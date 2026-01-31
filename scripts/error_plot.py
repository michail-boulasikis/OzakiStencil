import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

# Load data
df = pd.read_csv(sys.argv[1])

# --- Configuration ---
sns.set_theme(style="ticks", font_scale=1.0)
sns.set_context("talk")
palette = sns.color_palette("colorblind", 4)
markers = ["o", "s", "^", "D"]

# Reference Line Styles
ref_styles = {
    "float_ref": {"color": "#b8b8b8", "linestyle": "--", "linewidth": 2},
    "double_ref": {"color": "#787878", "linestyle": "-.", "linewidth": 2},
    "longdouble_ref": {"color": "#383838", "linestyle": ":", "linewidth": 2},
}
ref_names = {
    "float_ref": "Float Ref (Avg)",
    "double_ref": "Double Ref (Avg)",
    "longdouble_ref": "Long Double Ref (Avg)",
}
ref_order = ["float_ref", "double_ref", "longdouble_ref"]

# Get unique values
stencils = df["stencil"].unique()
timesteps = sorted(df["timesteps"].unique())

fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

for i, stencil in enumerate(stencils):
    ax = axes[i]
    data_stencil = df[df["stencil"] == stencil]

    # 1. Plot Main Lines (Ozaki)
    ozaki = data_stencil[data_stencil["type"] == "ozaki"]
    sns.lineplot(
        data=ozaki,
        x="slice_count",
        y="error",
        hue="timesteps",
        style="timesteps",
        palette=palette,
        markers=markers,
        dashes=False,
        linewidth=1,
        markersize=10,
        ax=ax,
        hue_order=timesteps,
        style_order=timesteps,
    )

    # 2. Plot Separate Reference Lines
    # Calculate mean error for each reference type (averaged over timesteps)
    refs = data_stencil[data_stencil["type"].isin(ref_styles.keys())]
    if not refs.empty:
        ref_means = refs.groupby("type")["error"].mean()

        for r_type in ref_order:
            if r_type in ref_means:
                val = ref_means[r_type]
                style = ref_styles[r_type]
                ax.axhline(
                    y=val,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=0.8,
                )

    # Formatting
    ax.set_yscale("log")
    ax.set_title(stencil, fontweight="bold", fontsize=14)
    ax.set_xlabel("Slice Count", fontweight="bold", fontsize=12)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    # Major ticks: Every power of 10
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))

    # Minor ticks: Between powers of 10 (e.g., 2e-1, 3e-1...)
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
    )

    # Ensure grid is behind data and minor ticks are enabled
    # ax.minorticks_on()
    # ax.set_axisbelow(True)

    # Grid lines
    ax.grid(True, which="major", linewidth=0.5, alpha=0.4, color="gray")
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.2, color="gray")

    if i == 0:
        ax.set_ylabel("RMSE (Log Scale)", fontweight="bold", fontsize=14)
    else:
        ax.set_ylabel("")

    if ax.get_legend():
        ax.get_legend().remove()

# Unified Legend
legend_elements = []

# Timesteps
for j, ts in enumerate(timesteps):
    line = Line2D(
        [0],
        [0],
        color=palette[j],
        marker=markers[j],
        linestyle="-",
        linewidth=3,
        markersize=8,
        label=f"T={ts}",
    )
    legend_elements.append(line)

# References
for r_type in ref_order:
    style = ref_styles[r_type]
    line = Line2D(
        [0],
        [0],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        label=ref_names[r_type],
    )
    legend_elements.append(line)

# Add Legend
fig.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=4,
    frameon=False,
    fontsize=14,
)

fig.suptitle("Stencil RMSE vs. Slice Count", fontsize=20, y=1.05)

plt.savefig("stencil_plot_detailed_refs.svg", bbox_inches="tight")
plt.show()
