import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

df = pd.read_csv(sys.argv[1])
df_melted = df.melt(
    id_vars=["brand", "model", "year"],
    value_vars=["int8", "fp16", "fp64"],
    var_name="precision",
    value_name="performance",
)

# "Poster" context scales everything up significantly.
# We add font_scale=1.1 to push it even further.
sns.set_context("talk", font_scale=1.1)
sns.set_style("whitegrid")

colors = sns.color_palette("colorblind", 3)
palette_dict = {"int8": colors[0], "fp16": colors[1], "fp64": colors[2]}

# Create figure. 20x9 is a robust aspect ratio for a full-width academic figure
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)


def plot_gpu_brand_bold(brand_name, ax, title):
    brand_data = df_melted[df_melted["brand"] == brand_name]

    # Massive lines and markers
    sns.lineplot(
        data=brand_data,
        x="year",
        y="performance",
        hue="precision",
        style="precision",
        markers=True,
        markersize=18,
        palette=palette_dict,
        linewidth=6,
        ax=ax,
        legend=False,
    )

    ax.set_yscale("log")
    # Bold title
    ax.set_title(title, fontsize=28, weight="bold", pad=25)
    # Bold labels
    ax.set_xlabel("Year", fontsize=24, weight="bold", labelpad=15)

    # Tick parameters: Make tick labels larger and bold
    ax.tick_params(axis="both", which="major", labelsize=20, width=2, length=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    # Grid: clearly visible but not overwhelming
    ax.grid(True, which="major", axis="y", ls="-", color="0.8", linewidth=1.5)
    ax.grid(False, axis="x")
    sns.despine(ax=ax)

    # Annotations: Bold and large
    for line_precision in ["int8", "fp16", "fp64"]:
        subset = brand_data[brand_data["precision"] == line_precision]
        for _, row in subset.iterrows():
            if pd.notna(row["performance"]):
                # Adjust offset slightly based on value to prevent overlap (simple heuristic)
                y_pos = row["performance"]
                ax.text(
                    row["year"],
                    y_pos,
                    f"  {row['model']}",
                    verticalalignment="center",
                    horizontalalignment="left",
                    fontsize=18,
                    fontweight="bold",
                    color="0.2",
                )


# Plot Nvidia
plot_gpu_brand_bold("nvidia", axes[0], "NVIDIA")
axes[0].set_ylabel("Perf. (TFLOPS or TOPS)", fontsize=24, weight="bold", labelpad=15)

# Plot AMD
plot_gpu_brand_bold("amd", axes[1], "AMD")
axes[1].set_ylabel("")

# Create a custom single legend for the whole figure at the top
# This prevents the legend from squishing the charts
from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0],
        [0],
        color=palette_dict["int8"],
        lw=6,
        marker="o",
        markersize=15,
        label="INT8",
    ),
    Line2D(
        [0],
        [0],
        color=palette_dict["fp16"],
        lw=6,
        marker="X",
        markersize=15,
        label="FP16",
    ),  # Note: seaborn default markers might differ, usually o, X, s. Checking defaults.
    # Actually seaborn style markers are mapped automatically. Let's grab handles from axes[0].
]
handles, labels = axes[0].get_legend_handles_labels()
# We need to recreate handles to ensure they match the size in the legend
# But simpler is to let figure legend handle it if we can grab them correctly.
# Let's manually build a robust legend to ensure visibility.

custom_lines = [
    Line2D([0], [0], color=palette_dict["int8"], lw=6, marker="o", markersize=15),
    Line2D([0], [0], color=palette_dict["fp16"], lw=6, marker="X", markersize=15),
    Line2D([0], [0], color=palette_dict["fp64"], lw=6, marker="s", markersize=15),
]
fig.legend(
    custom_lines,
    ["INT8", "FP16", "FP64"],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    ncol=3,
    fontsize=22,
    frameon=False,
)

plt.tight_layout()
plt.savefig(
    "data/figures/gpu_precision_trends.svg", bbox_inches="tight"
)  # bbox_inches='tight' helps with the external legend
plt.close()
