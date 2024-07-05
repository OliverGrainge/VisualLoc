import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the dataset
df = pd.read_csv("../results.csv")

# Define the parameters
dataset = "Nordlands"
device = "gpu"
batch_size = 1

# Show dataframe columns
print(df.columns)

# Ensure you are filtering by the right column for sparsity; here I assume it's called 'sparsity'
# and we're looking for rows where sparsity is closest to 30%
df["sparsity_diff"] = (df["sparsity"] - 30).abs()
df_closest_30 = df[
    df.groupby("agg_rate")["sparsity_diff"].transform(min) == df["sparsity_diff"]
]

# Check the DataFrame after filter


# Filter the relevant columns, assuming the memory columns are named as follows:
df_closest_30 = df_closest_30[
    [
        "method_name",
        f"{dataset}_matching_lat",  # Memory used by model on MapillarySLS dataset
        f"extraction_lat_{device}_bs{batch_size}",  # Memory used by the model itself
        "agg_rate",  # Aggregation pruning rates
    ]
]

df_closest_30[f"{dataset}_matching_lat"] = df[f"{dataset}_matching_lat"] * 1000

print(df_closest_30.head())
# Plotting the stacked bar chart
pivot_df = df_closest_30.pivot_table(
    index="agg_rate",
    values=[f"extraction_lat_{device}_bs{batch_size}", f"{dataset}_matching_lat"],
    aggfunc="sum",
)
pivot_df.plot(kind="bar", stacked=True)
plt.title("Latencye by Aggregation Pruning Rate")
plt.ylabel("Latency (Mb)")
plt.xlabel("Aggregation Pruning Rate")
plt.legend(title="Model Type")
plt.show()
