import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define parameters
aggregation = 0.75
dataset = "Pitts30k_Val"
device = "gpu"

# Read data from CSV files
df = pd.read_csv("../results.csv")

# Filter the dataframe based on the specified aggregation
df = df[
    ["method_name", f"{dataset}_total_{device}_lat_bs1", f"{dataset}_R1", "agg_rate"]
]
df = df[df["agg_rate"] == aggregation]

# Create a figure with normal axis
fig, ax = plt.subplots(figsize=(7, 3.5))

# Define color palette
color_palette = plt.get_cmap("viridis")(np.linspace(0, 1, df["method_name"].nunique()))
method_colors = dict(zip(df["method_name"].unique(), color_palette))

# Loop through each method and plot data along with a polynomial curve
for name, group in df.groupby("method_name"):
    # Plot raw data
    scatter = ax.scatter(
        group[f"{dataset}_total_{device}_lat_bs1"],
        group[f"{dataset}_R1"],
        label=name,
        color=method_colors[name],
    )

    # Fit a polynomial curve (2nd degree)
    coeffs = np.polyfit(
        group[f"{dataset}_total_{device}_lat_bs1"], group[f"{dataset}_R1"], 2
    )
    # Create a polynomial from coefficients
    p = np.poly1d(coeffs)

    # Create x values for the curve
    x = np.linspace(
        group[f"{dataset}_total_{device}_lat_bs1"].min(),
        group[f"{dataset}_total_{device}_lat_bs1"].max(),
        100,
    )
    y = p(x)

    # Plot the curve without adding to legend
    ax.plot(x, y, color=method_colors[name], linestyle="--", alpha=0.7)

# Custom legend labels
custom_labels = ["ConvAP ", "GeM", "MixVPR", "NetVLAD"]

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

ax.set_title("Pitts30k Val R@1 vs Total VPR latency (ms)")
ax.set_xlabel("Total System Latency (ms)")
ax.set_ylabel("Pitts30k Val R@1")

# Show plot
plt.savefig("total_latency.jpg", dpi=600)
plt.show()
