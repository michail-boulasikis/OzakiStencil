import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys

df = pd.read_csv(sys.argv[1])

main_data = df[df["Type"] == "ozaki"].copy()
ref_data = df[df["Type"] != "ozaki"].copy()

main_data["Iterations"] = main_data["Iterations"].astype(str)

plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")

sns.set_context("poster")
ax = sns.lineplot(
    data=main_data,
    x="Slices",
    y="Error",
    hue="Iterations",
    style="Iterations",
    markers=True,
    dashes=False,
)

ref_colors = {"float_ref": "red", "double_ref": "black"}

for ref_type in ref_data["Type"].unique():
    ref_value = ref_data[ref_data["Type"] == ref_type]["Error"].mean()

    plt.axhline(
        y=ref_value,
        color=ref_colors.get(ref_type, "gray"),
        linestyle="--",
        label=f"{ref_type}",
        alpha=0.7,
    )

ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.yscale("log")
plt.title("Error Convergence vs Slices (2D Heat Equation)")
plt.ylabel("RMSE (Log Scale)")
plt.legend()
plt.tight_layout()

plt.savefig("OzakiErrorsFigure.png")
plt.show()
