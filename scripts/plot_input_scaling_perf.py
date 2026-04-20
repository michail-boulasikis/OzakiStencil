#!/usr/bin/env python3
"""
Plot performance scaling line chart.

Usage:
    python plot_perf_line.py CSV_FILE
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys


COLORS = {
    "Ozaki + DSO2": "#87c55f",
    "Ozaki + DSO1": "#9eb9f3",
    "Ozaki": "#dcb0f2",
}


def main():
    # Quick check to ensure a file was passed
    if len(sys.argv) < 2:
        print("Usage: python plot_perf_line.py CSV_FILE")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    print(df)

    sns.set_theme(style="ticks", font_scale=1.5)
    sns.set_context("paper")
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot each variant
    variants = ["Ozaki + DSO1", "Ozaki + DSO2", "Ozaki"]
    for variant in variants:
        ax.plot(
            df["Input Size"],
            df[variant],
            marker="o",  # Add dots at the exact data points
            linewidth=2.5,
            markersize=6,
            label=variant,
            color=COLORS[variant],
        )

    # Configure X-axis to be logarithmic
    ax.set_xscale("log", base=2)
    ax.set_xticks(df["Input Size"])
    ax.set_xticklabels(df["Input Size"])

    # Set labels and title
    ax.set_xlabel("Input Size")
    ax.set_ylabel("Performance (GStPt/s)")
    ax.set_title("Performance Scaling with Input Size", pad=12, fontweight="bold")

    # Clean up the visual frame
    sns.despine(ax=ax)

    # Add legend (removed hardcoded fontsize so it inherits the larger font_scale)
    ax.legend(frameon=False, loc="upper left")

    # Layout and save
    plt.tight_layout()
    output_filename = "perf_scaling_line.pdf"
    fig.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Saved to {output_filename}")

    # Show the plot interactively
    plt.show()


if __name__ == "__main__":
    main()
