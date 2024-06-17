import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

method_name = "NetVLAD"

df = pd.read_csv("results.csv")

df = df[["method_name", "Pitts30k_Val_map_memory", "Pitts30k_Val_R1", "agg_rate"]]

df = df[df["method_name"] == method_name]

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="Pitts30k_Val_map_memory",
    y="Pitts30k_Val_R1",
    hue="agg_rate",
    style="agg_rate",
    markers=True,
    dashes=False,
)
plt.title(f"R1 vs Aggregation Pruning Rate for {method_name}")
plt.xlabel("Map Memory")
plt.ylabel("R1")
plt.legend(title="Aggregation Pruning Rate")
plt.grid(True)
plt.show()
