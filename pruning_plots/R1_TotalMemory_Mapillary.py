import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

agg_rate = 2.0
df = pd.read_csv("results.csv")
print(df.columns)

df = df[["method_name", "MapillarySLS_total_memory", "MapillarySLS_R1", "agg_rate"]]

df = df[df["agg_rate"] == agg_rate]

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="MapillarySLS_total_memory",
    y="MapillarySLS_R1",
    hue="method_name",
    style="method_name",
    markers=True,
    dashes=False,
)
plt.title(f"R1 vs System Memory Consumpation on MapillarySLS")
plt.xlabel("System Memory")
plt.ylabel("R1")
plt.legend(title="Aggregation Pruning Rate")
plt.grid(True)
plt.show()
