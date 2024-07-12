import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

aggregation = 0.75
dataset = "Pitts30k_Val"

df = pd.read_csv("../results.csv")

df = df[["method_name", "sparsity", f"model_memory", "agg_rate"]]
df = df[df["agg_rate"] == aggregation]

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="sparsity",
    y=f"model_memory",
    hue="method_name",
    style="method_name",
    markers=True,
    dashes=False,
)
plt.title(f"Sparsity vs model_memory {aggregation} Pruning Rate")
plt.xlabel("Sparsity")
plt.ylabel("Memory (mb))")
plt.legend(title="VPR Model")
plt.grid(True)
plt.show()
