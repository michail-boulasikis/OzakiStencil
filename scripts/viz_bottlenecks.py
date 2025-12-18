import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Configuration
INPUT_FILE = 'bottlenecks.csv'
OUTPUT_FILE = 'data/stencil_benchmark_chart_normalized.png'

def main():
    # 1. Validate Input
    if not os.path.exists(INPUT_FILE):
        print(f"Error: '{INPUT_FILE}' not found.")
        print("Please run the bash benchmark script first to generate the data.")
        sys.exit(1)

    # 2. Load Data
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if df.empty:
        print("Error: The CSV file is empty. No data to plot.")
        sys.exit(1)

    # 3. Data Preprocessing
    # Normalize execution time by problem size to see "Time per Element"
    # We keep this in nanoseconds (ns) because microseconds might be too small after division
    #df = df.where(df['radius'] < 8)
    df['normalized_ns'] = df['median_ns'] / (df['problem_size'] * df['problem_size'])

    # 4. Setup Plot Style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    # 5. Create Bar Chart
    # x = Stencil Radius (Independent Variable)
    # y = Normalized Time (Dependent Variable)
    # hue = Intent (Grouping Variable)
    ax = sns.barplot(
        data=df,
        x='problem_size',
        y='normalized_ns',
        hue='radius',
        palette='viridis',
        edgecolor='black',
        linewidth=1
    )

    # 6. Labeling
    plt.title('Stencil Kernel Efficiency (Normalized by Problem Size)', fontsize=16, weight='bold', pad=20)
    plt.xlabel('Problem Size (side)', fontsize=12)
    plt.ylabel('Median Time per Element (ns)', fontsize=12)
    
    # Improve Legend
    #plt.legend(title='Intent / Problem Size', title_fontsize='11', fontsize='10')
    plt.legend(title='Stencil Radius', title_fontsize='11', fontsize='10')

    # 7. Add Value Labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3, fontsize=9)

    # 8. Final Layout and Save
    plt.tight_layout()
    
    print(f"Saving chart to {OUTPUT_FILE}...")
    plt.savefig(OUTPUT_FILE, dpi=300)
    print("Done. Displaying plot...")
    plt.show()

if __name__ == "__main__":
    main()
