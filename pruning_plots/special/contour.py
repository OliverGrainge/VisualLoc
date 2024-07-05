import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# Load the dataset
df = pd.read_csv("../results.csv")

# Define the parameters
dataset = "MapillarySLS"
device = "cpu"
batch_size = 1
agg_rate = 1.5

# Assume these are the correct column names after verification
memory_col = f"{dataset}_total_memory"
r1_score_col = f"{dataset}_R1"
latency_col = f"{dataset}_total_{device}_lat_bs{batch_size}"

# Filter the relevant columns and for a specific method name
specific_method = (
    "MixVPR"  # Replace 'YourMethodName' with the actual method you are interested in
)
df = df[df["agg_rate"] == agg_rate]
df_filtered = df[df["method_name"] == specific_method][
    [memory_col, r1_score_col, latency_col]
]

# Grid the data
xi = np.linspace(df_filtered[memory_col].min(), df_filtered[memory_col].max(), 100)
yi = np.linspace(df_filtered[latency_col].min(), df_filtered[latency_col].max(), 100)
X, Y = np.meshgrid(xi, yi)
Z = griddata(
    (df_filtered[memory_col], df_filtered[latency_col]),
    df_filtered[r1_score_col],
    (X, Y),
    method="cubic",
)

# Plotting the contour
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=15, cmap="viridis")
plt.colorbar(label="Accuracy (R1 Score)")
plt.xlabel("Memory Consumption (MB)")
plt.ylabel("Latency (ms)")
plt.title(f"Contour Plot of Accuracy vs. Memory and Latency for {specific_method}")
plt.show()
