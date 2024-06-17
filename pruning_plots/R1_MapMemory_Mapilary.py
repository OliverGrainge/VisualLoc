import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

aggregation = 2.0

df = pd.read_csv("results.csv")


df = df[["method_name", "MapillarySLS_map_memory", "MapillarySLS_R1", "agg_rate"]]

df = df[df["agg_rate"] == aggregation]

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="MapillarySLS_map_memory",
    y="MapillarySLS_R1",
    hue="method_name",
    style="method_name",
    markers=True,
    dashes=False,
)
plt.title("R1 vs Backbone Pruning on MapillarySLS")
plt.xlabel("Map Memory")
plt.ylabel("R1")
plt.legend(title="VPR Model")
plt.grid(True)
plt.show()
