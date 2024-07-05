import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

aggregation = 1.5
dataset = "Pitts30k_Val"

df = pd.read_csv("../results.csv")

df = df[["method_name", "sparsity", f"{dataset}_R1", "agg_rate"]]
df = df[df["agg_rate"] == aggregation]

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="sparsity",
    y=f"{dataset}_R1",
    hue="method_name",
    style="method_name",
    markers=True,
    dashes=False,
)
plt.title(f"Sparsity vs {dataset} R@1 with {aggregation} Pruning Rate")
plt.xlabel("Sparsity")
plt.ylabel("R@1 (ms)")
plt.legend(title="VPR Model")
plt.grid(True)
plt.show()
