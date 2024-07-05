import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the dataset
df = pd.read_csv("../results.csv")

# Define the parameters
dataset = "Pitts30k_Val"
device = "cpu"
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
print(df_closest_30.head())

# Filter the relevant columns, assuming the memory columns are named as follows:
df_closest_30 = df_closest_30[
    [
        "method_name",
        f"{dataset}_map_memory",  # Memory used by model on MapillarySLS dataset
        "model_memory",  # Memory used by the model itself
        "agg_rate",  # Aggregation pruning rates
    ]
]

# Plotting the stacked bar chart
pivot_df = df_closest_30.pivot_table(
    index="agg_rate", values=[f"{dataset}_map_memory", "model_memory"], aggfunc="sum"
)
pivot_df.plot(kind="bar", stacked=True)
plt.title("Memory Usage by Aggregation Pruning Rate")
plt.ylabel("Memory (in units)")
plt.xlabel("Aggregation Pruning Rate")
plt.legend(title="Memory Type")
plt.show()
