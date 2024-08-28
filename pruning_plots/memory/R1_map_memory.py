import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define parameters
aggregation = 0.75
dataset = "Pitts30k_Val"

# Read data from CSV files
df = pd.read_csv("../results.csv")
baselines = pd.read_csv("../baselines2.csv")

# Replace the method_name entries
df["method_name"] = df["method_name"].replace("ResNet50_GeM", "GeM")

# Filter the dataframe based on the specified aggregation
df = df[["method_name", f"{dataset}_map_memory", f"{dataset}_R1", "agg_rate"]]
df = df[df["agg_rate"] == aggregation]

# Filter the baselines dataframe for the same dataset
baselines = baselines[["method_name", f"{dataset}_map_memory", f"{dataset}_R1"]]

# Create a figure with normal axis
fig, ax = plt.subplots(figsize=(7, 4))

# Define color palette
color_palette = plt.get_cmap("viridis")(np.linspace(0, 1, df["method_name"].nunique()))
method_colors = dict(zip(df["method_name"].unique(), color_palette))

# Plot data on the normal axis
for name, group in df.groupby("method_name"):
    ax.plot(
        group[f"{dataset}_map_memory"],
        group[f"{dataset}_R1"],
        label=name,
        marker="o",
        linestyle="-",
        color=method_colors[name],
    )

# Plot and annotate baseline data points
for name, group in baselines.groupby("method_name"):

    ax.scatter(
        group[f"{dataset}_map_memory"],
        group[f"{dataset}_R1"],
        label=f"{name} (baseline)",
        marker="x",
        color="black",
    )
    for index, row in group.iterrows():
        ax.annotate(
            name,
            (row[f"{dataset}_map_memory"], row[f"{dataset}_R1"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

# Add legend and labels

ax.legend()
ax.set_xlabel(f"{dataset}_map_memory")
ax.set_ylabel(f"{dataset}_R1")

# Show plot
plt.show()
