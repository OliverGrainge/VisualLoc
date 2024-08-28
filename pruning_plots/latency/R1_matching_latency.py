import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load data
df = pd.read_csv("../results.csv")

# Define parameters
aggregation = 0.0
dataset = "Pitts30k_Val"
device = "gpu"
batch_size = 1  # either 1 or 25

print(df.columns)

# Filter DataFrame based on the specified aggregation and necessary columns
df = df[
    [
        "method_name",
        f"{dataset}_matching_lat",
        f"{dataset}_R1",
        "agg_rate",
    ]
]
df = df[df["agg_rate"] == aggregation]

# Remove the last point from each group (method)
df = df.groupby("method_name").apply(lambda x: x.iloc[:-1]).reset_index(drop=True)

# Plotting
plt.figure(figsize=(7, 4))
sns.lineplot(
    data=df,
    x=f"{dataset}_matching_lat",
    y=f"{dataset}_R1",
    hue="method_name",
    style="method_name",
    markers=True,
    dashes=False,
)
plt.title(f"R@1 vs Extraction Latency for Pitts30k Val")
plt.xlabel("Latency (ms)", fontsize=12)
plt.ylabel("R@1", fontsize=12)
plt.legend(title="VPR Model", loc="center right")
plt.grid(True)
plt.savefig("R1_extraction_lat.jpg", dpi=500)
plt.show()
