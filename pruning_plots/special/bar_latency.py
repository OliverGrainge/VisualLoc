import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("../results.csv")

# Define the parameters
dataset = "Pitts30k_Val"
device = "gpu"
batch_size = 1

# Filter the dataframe
specific_method_name = "ResNet34_GeM"
specific_agg_rate = 0.5

df_filtered = df[
    (df["method_name"] == specific_method_name) & (df["agg_rate"] == specific_agg_rate)
]

# Filter the relevant columns
df_filtered = df_filtered[
    [
        "method_name",
        f"{dataset}_matching_lat",
        f"extraction_lat_{device}_bs{batch_size}",
        "agg_rate",
        "descriptor_dim",
    ]
]
df_filtered = df_filtered[df_filtered["agg_rate"] == specific_agg_rate]

print(df_filtered.head())

df_filtered = (
    df_filtered.groupby("agg_rate").apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(df_filtered))
width = 0.95

extraction_latency = df_filtered[f"extraction_lat_{device}_bs{batch_size}"]
matching_latency = df_filtered[f"{dataset}_matching_lat"]

ax.bar(x, extraction_latency, width, label="Extraction Latency", color="blue")
ax.bar(
    x,
    matching_latency,
    width,
    bottom=extraction_latency,
    label="Matching Latency",
    color="green",
)

plt.title(f"Latency for {specific_method_name} (Agg Rate: {specific_agg_rate})")
plt.ylabel("Latency (seconds)")
plt.xlabel("Row Index")
plt.legend(title="Latency Type")

plt.xticks(x, [str(i) for i in range(len(df_filtered))])
plt.grid(True, axis="y")

# Remove the top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Adjust the x-axis limits
plt.xlim(-0.5, len(df_filtered) - 0.5)

# Add descriptor dimensions on top of each bar
for i, (total_latency, dim) in enumerate(
    zip(extraction_latency + matching_latency, df_filtered["descriptor_dim"])
):
    ax.text(i, total_latency, f"{dim}", ha="center", va="bottom")

plt.tight_layout()
plt.show()
