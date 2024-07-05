import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the dataset
df = pd.read_csv("../results.csv")

# Define the parameters
dataset = "Pitts30k_Val"
device = "cpu"
batch_size = 1
agg_rate = 1.5

# Filter the relevant columns, fixed missing commas and added the correct column names
df = df[df["agg_rate"] == agg_rate]
df = df[
    [
        "method_name",
        f"{dataset}_total_memory",  # Check the correct column name for memory
        f"{dataset}_R1",  # Check the correct column name for R1 score
        f"{dataset}_total_{device}_lat_bs{batch_size}",  # Adjusted for the correct column name for latency
    ]
]


# Rename columns for better readability
df.columns = ["Method Name", "Memory Consumption (Mb)", "R1", "Latency"]

# Create a color palette for the methods
palette = sns.color_palette("magma", len(df["Method Name"].unique()))

# Plot the data
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=df,
    x="Latency",
    y="R1",
    size="Memory Consumption (Mb)",
    sizes=(40, 400),  # Adjust min and max size of bubbles
    hue="Method Name",
    palette=palette,
    alpha=0.6,
    edgecolors="w",
    linewidth=0.5,
)

# Add labels and title
plt.xlabel("Latency (ms)")
plt.ylabel("R1 Score")
plt.title("Latency vs R1 with Memory Consumption as Bubble Size")
plt.legend(
    title="Method Name",
    labelspacing=1.2,
    borderpad=1.5,
    loc="lower right",
    fontsize="small",
)

plt.grid(True)
plt.show()
