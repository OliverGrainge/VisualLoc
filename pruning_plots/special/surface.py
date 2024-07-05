import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
df = pd.read_csv("../results.csv")

# Define the parameters
dataset = "MapillarySLS"
device = "cpu"
batch_size = 1

# Assume these are the correct column names after verification
memory_col = f"{dataset}_total_memory"
r1_score_col = f"{dataset}_R1"
latency_col = f"{dataset}_total_{device}_lat_bs{batch_size}"

# Filter the relevant columns and for a specific method name
specific_method = (
    "MixVPR"  # Replace 'YourMethodName' with the actual method you are interested in
)
df_filtered = df[df["method_name"] == specific_method][
    [memory_col, r1_score_col, latency_col]
]

# Check and print the first few rows to ensure it's loaded correctly
print(df_filtered.head())

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(df_filtered[memory_col], df_filtered[latency_col], df_filtered[r1_score_col])

# Label the axes
ax.set_xlabel("Memory Consumption (MB)")
ax.set_ylabel("Latency (ms)")
ax.set_zlabel("Accuracy (R1 Score)")

# Set plot title
plt.title(f"3D Scatter Plot for {specific_method}")

# Show the plot
plt.show()
