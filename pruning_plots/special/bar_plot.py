import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
df = pd.read_csv("../results.csv")

# Define the parameters
dataset = "Pitts30k_Val"
device = "gpu"
batch_size = 1
method_name = "ResNet34_ConvAP"
agg_rate = 0.5

# Filter the dataframe based on the defined parameters
df = df[(df["method_name"] == method_name) & (df["agg_rate"] == agg_rate)]

# Ensure the necessary columns are present
extraction_column = f"extraction_lat_{device}_bs{batch_size}"
matching_column = f"{dataset}_matching_lat"
model_memory_column = "model_memory"
map_memory_column = f"{dataset}_map_memory"

palette1 = ["#e15759", "#76b7b2"]  # Colors for the first plot

fig, axs = plt.subplots(
    2, 1, figsize=(7, 6)
)  # Create a figure with 2 rows and 1 column

# Plot the first stacked bar chart
if extraction_column in df.columns and matching_column in df.columns:
    indices = range(len(df))
    axs[0].bar(
        indices,
        df[extraction_column],
        label="Extraction Latency (ms)",
        color=palette1[0],
    )
    axs[0].bar(
        indices,
        df[matching_column],
        bottom=df[extraction_column],
        label="Matching Latency (ms)",
        color=palette1[1],
    )
    axs[0].set_xlabel("Pruning Round")
    axs[0].set_ylabel("Latency (ms)", fontsize=11)
    axs[0].set_title(
        f"Extraction and Matching Latencies by Pruning Round (γ = {agg_rate})"
    )
    axs[0].legend()
    axs[0].set_xticks(indices)
    axs[0].set_xticklabels(indices)
else:
    print(
        f"Necessary columns {extraction_column} and/or {matching_column} are missing from the DataFrame"
    )

# Plot the second stacked bar chart
if model_memory_column in df.columns and map_memory_column in df.columns:
    indices = range(len(df))
    axs[1].bar(
        indices, df[model_memory_column], label="Model Memory (Mb)", color=palette1[0]
    )
    axs[1].bar(
        indices,
        df[map_memory_column],
        bottom=df[model_memory_column],
        label="Map Memory (Mb)",
        color=palette1[1],
    )
    axs[1].set_xlabel("Pruning Round")
    axs[1].set_ylabel("Memory Usage (Mb)", fontsize=11)
    axs[1].set_title(f"Model Memory and Map Memory by Pruning Round (γ = {agg_rate})")
    axs[1].legend()
    axs[1].set_xticks(indices)
    axs[1].set_xticklabels(indices)
else:
    print(
        f"Necessary columns {model_memory_column} and/or {map_memory_column} are missing from the DataFrame"
    )

plt.tight_layout()
plt.savefig("bar.jpg", dpi=600)
plt.show()
