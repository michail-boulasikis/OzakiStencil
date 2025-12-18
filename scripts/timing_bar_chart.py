import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re
import argparse


parser = argparse.ArgumentParser(description="Box stencil kernel generator and profiler")
parser.add_argument("--arch", type=str, choices=["A100", "H100"], required=True, help="The GPU used to carry out the experiments (e.g., A100)")
parser.add_argument("-i", "--input-file", type=str, help="Name of CSV file to open")
args = parser.parse_args()

# Read the data
df = pd.read_csv(f"data/reports/{args.arch}/{args.input_file}")

df['Framework'] = df.apply(lambda r: r['experiment'].split('_')[0], axis=1)
df['Stencil'] = df.apply(lambda r: r['experiment'].split('_')[1], axis=1)
df['GS/S'] = (8160 * 8160) / df['avg_ns']
df['GS/S'] = df.apply(lambda r: r['GS/S'] / 4 if r['Framework'] == "tcstencil" else r['GS/S'], axis=1)
df['GS/S'] = df.apply(lambda r: r['GS/S'] * 3 if r['Framework'] in ["convstencil", "lorastencil"] and r['Stencil'] in ['box2d1r', 'star2d1r'] else r['GS/S'], axis=1)

# Sort for consistent plotting
df = df.sort_values(by=['Stencil', 'Framework'])

# Using CatPlot to create a FacetGrid (Separating Box vs Star)
g = sns.catplot(
    data=df, 
    x='Stencil', 
    y='GS/S', 
    hue='Framework', 
    kind='bar',
    height=5, 
    aspect=1.2,
)

# Formatting
g.set_axis_labels("Stencil", "GStencils/s")
g.set_titles("{col_name} Stencil")


for ax in g.axes.flat:
    ax.grid(color='gainsboro', linestyle='--', linewidth=1.5)
    ax.set_axisbelow(True)

# Set Log Scale on Y-axis (crucial due to large variance in values)
#for ax in g.axes.flat:
#    ax.set_yscale('log')

plt.subplots_adjust(top=0.85)
g.fig.suptitle(f'Performance Comparison: Stencil Frameworks ({args.arch} GPU)')
plt.savefig(f'data/stencil_comparison_{args.arch}.png')
plt.show()
