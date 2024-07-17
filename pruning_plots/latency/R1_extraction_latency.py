import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Define parameters
aggregation = 0.75
dataset = "Pitts30k_Val"
device = "gpu"

# Read data from CSV files
df = pd.read_csv("../results.csv")

# Filter the dataframe based on the specified aggregation
df = df[["method_name", f"extraction_lat_{device}_bs1", f"{dataset}_R1", "agg_rate"]]
df = df[df["agg_rate"] == aggregation]

# Create a figure with normal axis
fig, ax = plt.subplots(figsize=(7, 4))

# Define color palette
method_names = df["method_name"].unique()
color_palette = sns.color_palette("viridis", len(method_names))
method_colors = dict(zip(method_names, color_palette))

# Plot data with a regression line for each method
for name in method_names:
    method_data = df[df["method_name"] == name]
    sns.scatterplot(
        data=method_data,
        x=f"extraction_lat_{device}_bs1",
        y=f"{dataset}_R1",
        ax=ax,
        label=name,
        color=method_colors[name],
    )
    sns.regplot(
        data=method_data,
        x=f"extraction_lat_{device}_bs1",
        y=f"{dataset}_R1",
        ax=ax,
        scatter=False,
        color=method_colors[name],
        ci=None,
    )

# Custom legend labels
custom_labels = ["ConvAP", "GeM", "MixVPR", "NetVLAD"]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, custom_labels)

# Add labels and title
ax.set_title("Pitts30k Val R@1 vs Total VPR Latency")
ax.set_xlabel("Extraction Latency (ms)")
ax.set_ylabel("Pitts30k Val R@1")

# Show plot
plt.show()
