import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Define parameters
aggregation = 0.75
dataset = "Pitts30k_Val"

# Read data from CSV files
df = pd.read_csv("../results.csv")

# Replace the method_name entries
df["method_name"] = df["method_name"].replace("ResNet50_GeM", "GeM")

# Filter the dataframe based on the specified aggregation
df = df[["method_name", f"{dataset}_total_memory", f"{dataset}_R1", "agg_rate"]]
df = df[df["agg_rate"] == aggregation]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(7, 3.5))

# Define color palette
color_palette = sns.color_palette("viridis", df["method_name"].nunique())
method_colors = dict(zip(df["method_name"].unique(), color_palette))

# Loop through each method and plot data along with a linear fit
for name, group in df.groupby("method_name"):
    # Fit a linear model (degree 1 polynomial)
    coeffs = np.polyfit(group[f"{dataset}_total_memory"], group[f"{dataset}_R1"], 1)
    print(f"Method: {name}, Coefficients: {coeffs}")

    # Plot raw data with linear regression line
    sns.regplot(
        ax=ax,
        x=group[f"{dataset}_total_memory"],
        y=group[f"{dataset}_R1"],
        label=name,
        color=method_colors[name],
        scatter_kws={"s": 50},  # size of scatter points
        line_kws={"color": method_colors[name], "alpha": 0.7, "linestyle": "--"},
    )

# Custom legend labels
custom_labels = ["ConvAP", "GeM", "MixVPR", "NetVLAD"]

# Create custom legend handles
handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=method_colors[name],
        markersize=10,
        label=label,
    )
    for name, label in zip(df["method_name"].unique(), custom_labels)
]

# Add legend with custom handles
ax.legend(handles=handles, title="Method")

ax.set_title(f"Pitts30k Val R@1 vs Total VPR Memory (γ = {aggregation})")
ax.set_xlabel("Total System Memory (Mb)")
ax.set_ylabel("Pitts30k Val R@1")
plt.subplots_adjust(bottom=0.15)
# Show plot
plt.savefig("total_mem.jpg", dpi=600)
plt.show()
