import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Constants and data loading
aggregation = 2.0
dataset = "Pitts30k_Val"
method = "ConvAP"
device = "gpu"
batch_size = 1  # either 1 or 25

df = pd.read_csv("../results.csv")
print(df.columns)

# Filter and keep necessary columns
df = df[
    [
        "method_name",
        f"{dataset}_total_memory",
        f"{dataset}_total_gpu_lat_bs1",
        "agg_rate",
        f"{dataset}_R1",
    ]
]

# Define the method you want to plot
method_of_interest = (
    method  # Replace 'YourMethodName' with the method you're interested in
)

# Filter data for a specific method
df_filtered = df[df["method_name"] == "ConvAP"]
agg_rates_of_interest = [0.0, 0.5, 1.0, 1.5]
df_filtered = df_filtered[df_filtered["agg_rate"].isin(agg_rates_of_interest)]


# Set up the figure and axes
fig, axs = plt.subplots(2, 1, figsize=(6, 5))  # 2 rows, 1 column

# First subplot: Recall vs. Memory
sns.lineplot(
    ax=axs[0],
    data=df_filtered,
    x=f"{dataset}_total_memory",
    y=f"{dataset}_R1",
    hue="agg_rate",
    marker="o",
)
axs[0].set_title(f"Recall vs Memory for {method_of_interest}")
axs[0].set_xlabel("Memory (Mb)")
axs[0].set_ylabel("R@1")
axs[0].legend(title="gamma γ")

# Second subplot: Recall vs. Latency
sns.lineplot(
    ax=axs[1],
    data=df_filtered,
    x=f"{dataset}_total_gpu_lat_bs1",
    y=f"{dataset}_R1",
    hue="agg_rate",
    marker="o",
)
axs[1].set_title(f"Recall vs Latency for {method_of_interest}")
axs[1].set_xlabel("Latency (ms)")
axs[1].set_ylabel("R@1")
axs[1].legend(title="gamma γ")

# Display the plots
plt.tight_layout()
plt.savefig("gamma.jpg", dpi=500)
plt.show()
