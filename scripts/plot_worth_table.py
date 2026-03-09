#!/usr/bin/env python3
"""
Plot OzakiStencil winner table from ozaki_worth_table.sh TSV output.

For each cell (data slices s, stencil M value):
  - Determine the equivalent naive precision (FP32 ≡ s=5, FP64 ≡ s=9, FP128 ≡ s=17)
  - If Ozaki Direct GStPt/s > naive GStPt/s  →  green  ("Ozaki wins")
  - Otherwise                                →  red shade matching the naive precision

Cell text shows the winning algorithm and the speedup ratio
(winner GStPt/s / loser GStPt/s, so ratio > 1 always means "winner is that many times faster").

Usage:
    python scripts/plot_worth_table.py <tsv_file> [-o output.png]
"""

import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import NullLocator

# ── Precision equivalences ────────────────────────────────────────────────────
FP32_SLICES  = 5
FP64_SLICES  = 9
FP128_SLICES = 17

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "FP32" : "#e05050",   # light red   (low  precision region)
    "FP64" : "#b01515",   # medium red  (mid  precision region)
    "FP128": "#5a0000",   # dark red    (high precision region)
    "Ozaki": "#2a9e4f",   # green
    "N/A"  : "#999999",   # grey
}
TEXT_COLOR = "white"


def equivalent_naive(s: int) -> str:
    if s <= FP32_SLICES:  return "FP32"
    if s <= FP64_SLICES:  return "FP64"
    return "FP128"


# ── Parsing ───────────────────────────────────────────────────────────────────

def _try_float(v: str):
    try:
        return float(v)
    except ValueError:
        return None


def parse_tsv(path: str):
    baselines: dict[str, float | None] = {}
    m_cols: list[str] = []
    rows: list[tuple[int, dict[str, float | None]]] = []

    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("#"):
                m = re.search(
                    r"baseline_fp32=(\S+)\s+baseline_fp64=(\S+)\s+baseline_fp128=(\S+)",
                    line,
                )
                if m:
                    baselines["FP32"]  = _try_float(m.group(1))
                    baselines["FP64"]  = _try_float(m.group(2))
                    baselines["FP128"] = _try_float(m.group(3))
            elif line.startswith("slices"):
                m_cols = line.split("\t")[1:]
            elif line:
                parts = line.split("\t")
                s = int(parts[0])
                vals = {
                    col: _try_float(parts[i + 1]) if i + 1 < len(parts) else None
                    for i, col in enumerate(m_cols)
                }
                rows.append((s, vals))

    return baselines, m_cols, rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot OzakiStencil winner table"
    )
    parser.add_argument("tsv_file", help="TSV produced by ozaki_worth_table.sh")
    parser.add_argument("-o", "--output", default=None, help="Save to file instead of showing")
    args = parser.parse_args()

    baselines, m_cols, rows = parse_tsv(args.tsv_file)
    n_rows = len(rows)
    n_cols = len(m_cols)
    slice_list = [s for s, _ in rows]

    # ── Build per-cell data ───────────────────────────────────────────────────
    cell_color = np.empty((n_rows, n_cols), dtype=object)
    cell_label = np.empty((n_rows, n_cols), dtype=object)
    cell_ratio = np.full((n_rows, n_cols), np.nan)   # ratio > 1 means winner is faster

    for i, (s, vals) in enumerate(rows):
        ref_algo = equivalent_naive(s)
        ref_gstp = baselines.get(ref_algo)
        for j, col in enumerate(m_cols):
            oz = vals.get(col)
            if oz is None or ref_gstp is None or ref_gstp == 0:
                cell_color[i, j] = COLORS["N/A"]
                cell_label[i, j] = "N/A"
            elif oz > ref_gstp:
                ratio = oz / ref_gstp
                cell_ratio[i, j] = ratio
                cell_color[i, j] = COLORS["Ozaki"]
                cell_label[i, j] = f"Ozaki\n×{ratio:.2f}"
            else:
                ratio = ref_gstp / oz
                cell_ratio[i, j] = ratio
                cell_color[i, j] = COLORS[ref_algo]
                cell_label[i, j] = f"{ref_algo}\n×{ratio:.2f}"

    # ── Figure setup ─────────────────────────────────────────────────────────
    CELL_W = 2.1     # axes-unit width of one cell
    CELL_H = 0.70    # axes-unit height of one cell
    LEFT_ANN  = 0.80  # space for precision-region labels (left of grid)
    RIGHT_ANN = 0.15  # right padding inside axes
    AX_W = LEFT_ANN + n_cols * CELL_W + RIGHT_ANN
    AX_H = n_rows * CELL_H

    # We use a single axes that spans the whole figure; margins are set manually.
    MARGIN_LEFT   = 0.55   # figure-fraction
    MARGIN_RIGHT  = 0.20
    MARGIN_TOP    = 0.18
    MARGIN_BOTTOM = 0.10
    LEGEND_EXTRA  = 2.6    # inches added to the right for legend

    fig_w = AX_W + LEGEND_EXTRA
    fig_h = AX_H + 1.4   # top + bottom padding in inches (approximate)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(
        left=MARGIN_LEFT / fig_w,
        right=1.0 - MARGIN_RIGHT / fig_w,
        top=1.0 - MARGIN_TOP / fig_h,
        bottom=MARGIN_BOTTOM / fig_h,
    )

    # Axes coordinate system: x in [0, n_cols*CELL_W], y in [0, n_rows*CELL_H]
    ax.set_xlim(0, n_cols * CELL_W)
    ax.set_ylim(0, n_rows * CELL_H)
    ax.invert_yaxis()  # row 0 at top

    # ── Draw cells ───────────────────────────────────────────────────────────
    for i in range(n_rows):
        for j in range(n_cols):
            x0 = j * CELL_W
            y0 = i * CELL_H
            rect = plt.Rectangle(
                (x0, y0), CELL_W, CELL_H,
                color=cell_color[i, j], ec="white", lw=1.0,
            )
            ax.add_patch(rect)
            ax.text(
                x0 + CELL_W * 0.5, y0 + CELL_H * 0.5,
                cell_label[i, j],
                ha="center", va="center",
                fontsize=8.0, color=TEXT_COLOR, fontweight="bold",
                linespacing=1.35,
            )

    # ── Precision-region separator lines & left-side labels ──────────────────
    # Find row indices where the region changes
    region_breaks: list[int] = []   # row index just BEFORE a boundary line
    prev_region = equivalent_naive(slice_list[0])
    for i, s in enumerate(slice_list):
        cur = equivalent_naive(s)
        if cur != prev_region and i > 0:
            region_breaks.append(i)
            prev_region = cur

    for i_break in region_breaks:
        y_line = i_break * CELL_H
        ax.axhline(y_line, color="white", lw=2.5, ls="-", zorder=5)

    # Identify contiguous region spans for left-side labels
    region_spans: list[tuple[int, int, str]] = []  # (row_start, row_end_excl, algo)
    start = 0
    prev_r = equivalent_naive(slice_list[0])
    for i, s in enumerate(slice_list[1:], 1):
        cur = equivalent_naive(s)
        if cur != prev_r:
            region_spans.append((start, i, prev_r))
            start, prev_r = i, cur
    region_spans.append((start, n_rows, prev_r))

    for r_start, r_end, algo in region_spans:
        y_mid = (r_start + r_end) / 2 * CELL_H
        ax.text(
            -0.18, y_mid, algo,
            ha="right", va="center",
            fontsize=9, fontweight="bold", color=COLORS[algo],
            transform=ax.transData,
        )

    # ── Axes ticks & labels ───────────────────────────────────────────────────
    # X: M-value column headers
    ax.set_xticks([(j + 0.5) * CELL_W for j in range(n_cols)])
    ax.set_xticklabels(m_cols, fontsize=10)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="both", bottom=False, top=False)

    # Y: slice count; highlight precision-equivalent rows
    PREC_ROWS = {FP32_SLICES: "FP32", FP64_SLICES: "FP64", FP128_SLICES: "FP128"}
    ax.set_yticks([(i + 0.5) * CELL_H for i in range(n_rows)])
    y_labels = [str(s) for s in slice_list]
    ax.set_yticklabels(y_labels, fontsize=8.5)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="y", which="both", left=False, right=False)

    # Colour the tick labels of the precision-equivalent rows
    for i, s in enumerate(slice_list):
        if s in PREC_ROWS:
            ax.get_yticklabels()[i].set_color(COLORS[PREC_ROWS[s]])
            ax.get_yticklabels()[i].set_fontweight("bold")

    ax.set_xlabel("Stencil M value", fontsize=11, labelpad=6)
    ax.set_ylabel("Data slices  (s)", fontsize=11, labelpad=6)

    # Axes border
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Title ─────────────────────────────────────────────────────────────────
    def _fmt(v):
        return f"{v:.2f}" if v is not None else "N/A"

    title_lines = (
        "OzakiStencil Winner Table\n"
        f"FP32 = {_fmt(baselines.get('FP32'))} GStPt/s  ·  "
        f"FP64 = {_fmt(baselines.get('FP64'))} GStPt/s  ·  "
        f"FP128 = {_fmt(baselines.get('FP128'))} GStPt/s"
    )
    ax.set_title(title_lines, fontsize=11, pad=10, loc="center")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=COLORS["Ozaki"], label="Ozaki Direct wins"),
        mpatches.Patch(color=COLORS["FP32"],
                       label=f"FP32 wins  (s ≤ {FP32_SLICES})"),
        mpatches.Patch(color=COLORS["FP64"],
                       label=f"FP64 wins  ({FP32_SLICES} < s ≤ {FP64_SLICES})"),
        mpatches.Patch(color=COLORS["FP128"],
                       label=f"FP128 wins  (s > {FP64_SLICES})"),
        mpatches.Patch(color=COLORS["N/A"], label="N/A"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=9, framealpha=0.92, edgecolor="#cccccc",
        title="×ratio = winner / loser\nGStPt/s",
        title_fontsize=8,
    )

    # ── Save / show ───────────────────────────────────────────────────────────
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
