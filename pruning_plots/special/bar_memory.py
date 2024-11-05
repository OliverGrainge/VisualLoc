import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("../results.csv")

# Define the parameters
dataset = "Pitts30k_Val"
device = "cpu"
batch_size = 1

# Filter to a specific method_name and agg_rate
specific_method_name = "ResNet34_GeM"
specific_agg_rate = 0.5  # Replace with the agg_rate you're interested in

df_filtered = df[
    (df["method_name"] == specific_method_name) & (df["agg_rate"] == specific_agg_rate)
]

# Filter the relevant columns
df_filtered = df_filtered[
    [
        "method_name",
        f"{dataset}_map_memory",  # Memory used by model on dataset
        "model_memory",  # Memory used by the model itself
        "agg_rate",  # Aggregation pruning rates
    ]
]

# Plotting each row as a separate bar, lined up next to each other without space
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(df_filtered))
width = 0.95  # Set width to 1 to remove spacing between bars

ax.bar(x, df_filtered[f"{dataset}_map_memory"], width, label="Map Memory", color="blue")
ax.bar(
    x,
    df_filtered["model_memory"],
    width,
    bottom=df_filtered[f"{dataset}_map_memory"],
    label="Model Memory",
    color="green",
)

plt.title(f"Memory Usage for {specific_method_name} (Agg Rate: {specific_agg_rate})")
plt.ylabel("Memory (in units)")
plt.xlabel("Row Index")
plt.legend(title="Memory Type")

# Set x-ticks to be the row indices
plt.xticks(x, [str(i) for i in range(len(df_filtered))])

plt.grid(True, axis="y")
plt.tight_layout()

# Remove the top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Adjust the x-axis limits to remove extra space on the sides
plt.xlim(-0.5, len(df_filtered) - 0.5)

plt.show()
