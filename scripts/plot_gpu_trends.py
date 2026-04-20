import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np

df = pd.read_csv("data/gpu_prec_trends.csv")

COLORS = {
    "INT8": "#66c5cc",
    "FP16": "#f6cf71",
    "FP64": "#f89c74",
}

MARKERS = {
    "INT8": "o",
    "FP16": "s",
    "FP64": "^",
}

METRICS = ["INT8", "FP16", "FP64"]
LABELS = {"INT8": "INT8 (TOPS)", "FP16": "FP16 (TFLOPS)", "FP64": "FP64 (TFLOPS)"}

# Per-GPU label offsets: (dx years, dy in log10 units)
OFFSETS = {
    # NVIDIA
    "V100": (-0.25, 0.22),
    "A100": (0.08, 0.20),
    "H100": (-0.40, 0.20),
    "B100": (0.08, -0.32),
    # AMD
    "MI50": (-0.25, 0.22),
    "MI100": (0.08, 0.20),
    "MI250X": (-0.60, 0.20),
    "MI300X": (-0.50, 0.22),
    "MI350X": (-0.60, -0.20),
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 2.2,
        "lines.markersize": 6,
    }
)

sns.set_theme(style="white", font_scale=1.0)
# Re-apply rcParams after seaborn reset
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 2.2,
        "lines.markersize": 6,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True, layout="constrained")
fig.suptitle("GPU Performance Trends (TOPS / TFLOPS)", fontsize=10, fontweight="bold")

for ax, company in zip(axes, ["NVIDIA", "AMD"]):
    sub = df[df["Company"] == company].sort_values("Year")

    for metric in METRICS:
        valid = sub.dropna(subset=[metric])
        if valid.empty:
            continue

        years = valid["Year"].values
        vals = valid[metric].values

        ax.plot(
            years,
            vals,
            color=COLORS[metric],
            linewidth=2.2,
            marker=MARKERS[metric],
            markersize=6,
            label=LABELS[metric],
            zorder=3,
        )

        for _, row in valid.iterrows():
            dx, dy = OFFSETS.get(row["Name"], (0.08, 0.20))
            y_ann = 10 ** (np.log10(row[metric]) + dy)
            ax.text(
                row["Year"] + dx,
                y_ann,
                row["Name"],
                fontsize=7.5,
                # color=COLORS[metric],
                va="center",
                fontweight="semibold",
            )

    ax.set_yscale("log")
    ax.set_title(company, fontsize=10, fontweight="bold")
    ax.set_xlabel("Year", fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    for yval in [10, 100, 1000]:
        ax.axhline(yval, color="gray", linewidth=0.7, linestyle="--", zorder=1)

    if ax is axes[0]:
        ax.set_ylabel("TFLOPS or TOPS", fontsize=9)

# Shared legend: collect handles from the first axes (all three metrics are plotted there)
handles, labels_ = axes[0].get_legend_handles_labels()
seen = {}
for h, lbl in zip(handles, labels_):
    seen.setdefault(lbl, h)
fig.legend(
    seen.values(),
    seen.keys(),
    loc="outside lower center",
    ncol=len(seen),
    framealpha=0.85,
    borderpad=0.4,
    labelspacing=0.3,
    columnspacing=1.0,
    fontsize=8,
)
plt.savefig("scripts/gpu_prec_trends.png", dpi=200, bbox_inches="tight")
print("Saved scripts/gpu_prec_trends.png")
plt.show()
