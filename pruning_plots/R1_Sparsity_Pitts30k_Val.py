import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

aggregation = 0.5

df = pd.read_csv("results.csv")


df = df[["method_name", "sparsity", "Pitts30k_Val_R1", "agg_rate"]]
df = df[df["agg_rate"] == aggregation]

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="sparsity",
    y="Pitts30k_Val_R1",
    hue="method_name",
    style="method_name",
    markers=True,
    dashes=False,
)
plt.title("R1 vs Backbone Pruning on Pitts30k Val")
plt.xlabel("Sparsity")
plt.ylabel("R1")
plt.legend(title="VPR Model")
plt.grid(True)
plt.show()
