import matplotlib.pyplot as plt
import pandas as pd
from brokenaxes import brokenaxes
import matplotlib.gridspec as gridspec
import numpy as np

aggregation = 1.5
dataset = "Pitts30k_Val"

df = pd.read_csv("../results.csv")
baselines = pd.read_csv("../baselines.csv")

# Replace the method_name entries
df["method_name"] = df["method_name"].replace("ResNet50_GeM", "GeM")

df = df[["method_name", f"{dataset}_total_memory", f"{dataset}_R1", "agg_rate"]]
df = df[df["agg_rate"] == aggregation]

# Create a figure with broken x-axis
fig = plt.figure(figsize=(7, 4))
gs = gridspec.GridSpec(1, 1)
bax = brokenaxes(xlims=((0, 400), (600, 700)), subplot_spec=gs[0], d=0.05)

# Manually plotting
color_palette = plt.get_cmap("viridis")(np.linspace(0, 1, df["method_name"].nunique()))
method_colors = dict(zip(df["method_name"].unique(), color_palette))

for name, group in df.groupby("method_name"):
    bax.plot(
        group[f"{dataset}_total_memory"],
        group[f"{dataset}_R1"],
        label=name,
        marker="o",
        linestyle="-",
        color=method_colors[name],
    )

# Add baseline points
for i in range(len(baselines)):
    row = baselines.iloc[i]
    memory = float(row[f"{dataset}_total_memory"])
    name = str(row["method_name"])
    if "dino" not in name.lower():
        continue
    r1 = row[f"{dataset}_R1"]
    bax.plot(memory, r1, "o", color="red")  # Red dot for baseline
    bax.text(memory - 5, r1 - 0.02, name, ha="center", va="bottom", fontsize=11)

bax.set_title(f"R@1 vs Memory Consumption for Pitts30k Val")
bax.set_xlabel("Memory (Mb)", fontsize=13, labelpad=20)
bax.set_ylabel("R@1", fontsize=13, labelpad=42)
bax.grid(True)
bax.legend(title="VPR Model", loc="center right")
plt.savefig("total_mem.jpg", dpi=500, format="jpg")
plt.show()
