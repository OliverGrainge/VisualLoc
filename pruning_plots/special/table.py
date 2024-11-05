import pandas as pd

# Constants
dataset = "Pitts30k_Val"
method = "ConvAP"

# Load data
df = pd.read_csv("../results.csv")

# Define necessary columns
columns_needed = [
    "method_name",
    "agg_rate",
    f"extraction_lat_gpu_bs1",
    f"{dataset}_matching_lat",
    f"{dataset}_map_memory",
    f"model_memory",
    f"{dataset}_R1",
    "sparsity",
]

# Filter and keep necessary columns
df = df[columns_needed]

# Rename columns
rename_columns = {
    "method_name": "Method",
    "agg_rate": "Gamma",
    "extraction_lat_gpu_bs1": "Extraction Latency",
    f"{dataset}_matching_lat": "Matching Latency",
    f"{dataset}_map_memory": "Map Memory",
    f"model_memory": "Model Memory",
    f"{dataset}_R1": "R@1",
}

df.rename(columns=rename_columns, inplace=True)

# Replace 'ResNet50_GeM' with 'GeM'
df["Method"] = df["Method"].replace("ResNet50_GeM", "GeM")

# Convert Matching Latency from seconds to milliseconds
df["Matching Latency"] *= 1000


# Filter by closest sparsity to 25%
def filter_closest_sparsity(group):
    closest_row = group.iloc[(group["sparsity"] - 25).abs().argsort()[:1]]
    return closest_row


filtered_df = (
    df.groupby(["Method", "Gamma"])
    .apply(filter_closest_sparsity)
    .reset_index(drop=True)
)

# Generate LaTeX code with column lines
column_format = "|l|r|r|r|r|r|r|r|"
latex_table = filtered_df.to_latex(
    index=False,
    column_format=column_format,
    caption="Filtered Data Closest to 25% Sparsity",
    label="tab:sparsity_data",
    float_format="%.2f",
)

print(latex_table)
