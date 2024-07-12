import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

aggregation = 00
dataset = "Pitts30k_Val"
device = "gpu"
batch_size = 1  # either 1 or 25

df = pd.read_csv("../results.csv")
print(df.columns)

df = df[
    [
        "method_name",
        f"{dataset}_total_{device}_lat_bs{batch_size}",
        f"sparsity",
        "agg_rate",
    ]
]
df = df[df["agg_rate"] == aggregation]


# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="sparsity",
    y=f"{dataset}_total_{device}_lat_bs{batch_size}",
    hue="method_name",
    style="method_name",
    markers=True,
    dashes=False,
)
plt.title(f"Sparsity vs Total Latency (Batch Size: {batch_size} device: {device})")
plt.xlabel("Sparsity")
plt.ylabel("Latency (ms)")
plt.legend(title="VPR Model")
plt.grid(True)
plt.show()
