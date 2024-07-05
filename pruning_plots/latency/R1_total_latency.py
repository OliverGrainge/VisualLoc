import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

aggregation = 2.0
dataset = "Pitts30k_Val"
device = "gpu"
batch_size = 1  # either 1 or 25

df = pd.read_csv("../results.csv")
print(df.columns)

df = df[
    [
        "method_name",
        f"{dataset}_total_{device}_lat_bs{batch_size}",
        f"{dataset}_R1",
        "agg_rate",
    ]
]
df = df[df["agg_rate"] == aggregation]


# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x=f"{dataset}_total_{device}_lat_bs{batch_size}",
    y=f"{dataset}_R1",
    hue="method_name",
    style="method_name",
    markers=True,
    dashes=False,
)
plt.title(
    f"Total Latency vs R@1 for {dataset} (Batch Size: {batch_size} device: {device})"
)
plt.xlabel("Latency (ms)")
plt.ylabel("R@1")
plt.legend(title="VPR Model")
plt.grid(True)
plt.show()
