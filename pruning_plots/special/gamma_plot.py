import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Constants and data loading
aggregation = 0.0
dataset = "Pitts30k_Val"
method = "ResNet34_ConvAP"
device = "gpu"
batch_size = 1  # either 1 or 25

df = pd.read_csv("../results.csv")


# Filter and keep necessary columns
df = df[
    [
        "method_name",
        f"{dataset}_total_memory",
        f"{dataset}_total_gpu_lat_bs1",
        "agg_rate",
        f"{dataset}_R1",
    ]
]

# Define the method you want to plot
method_of_interest = method

# Filter data for a specific method
df_filtered = df[df["method_name"] == method_of_interest]
agg_rates_of_interest = [0.0, 0.25, 0.5, 0.75]
df_filtered = df_filtered[df_filtered["agg_rate"].isin(agg_rates_of_interest)]

# Set up the figure and axes
fig, axs = plt.subplots(2, 1, figsize=(6, 5.2))  # 2 rows, 1 column

# Define color palette
palette = sns.color_palette("viridis", len(agg_rates_of_interest))

# Replace underscores with spaces for method title
method_title = method.replace("_", " ")

# Plot and fit each subplot
i = 0
for ax, feature, title in zip(
    axs,
    [f"{dataset}_total_memory", f"{dataset}_total_gpu_lat_bs1"],
    ["Recall vs Memory", "Recall vs Latency"],
):
    sns.scatterplot(
        ax=ax,
        data=df_filtered,
        x=feature,
        y=f"{dataset}_R1",
        hue="agg_rate",
        marker="o",
        palette=palette,
    )
    ax.set_title(f"{title} for {method_title}")
    ax.set_xlabel(
        feature.split("_")[-2].capitalize() + " (Mb)" if "memory" in feature else "(ms)"
    )
    if i == 0:
        ax.set_xlabel("Memory Usage (Mb)")
    else:
        ax.set_xlabel("Latency (ms)")

    i += 1
    ax.set_ylabel("R@1")
    ax.legend(title="gamma γ")

    # Fit polynomial curves
    for (gamma, group), color in zip(df_filtered.groupby("agg_rate"), palette):
        # Fit a 2nd degree polynomial
        coeffs = np.polyfit(group[feature], group[f"{dataset}_R1"], 2)
        p = np.poly1d(coeffs)
        # Generate x values
        x_vals = np.linspace(group[feature].min(), group[feature].max(), 100)
        y_vals = p(x_vals)
        # Plot the curve
        ax.plot(x_vals, y_vals, color=color, linestyle="--")

# Display the plots
plt.tight_layout()
plt.savefig("gamma_plot.jpg", dpi=600)
plt.show()
