import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def pruning_schedule(end, epoch: int, cumulative=False):
    start = 0
    max_epochs = 30
    pruning_freq = 2

    if cumulative:
        if epoch == 0:
            rate = start + (end - start) * (pruning_freq / (max_epochs))
            return rate
        elif epoch >= max_epochs:
            curr = start + (end - start) * (epoch / (max_epochs))
            prev = start + (end - start) * ((epoch - pruning_freq) / (max_epochs))
            return curr - prev
    else:
        if epoch == 0:
            return 0  # No pruning at the very start if not cumulative
        elif epoch >= max_epochs:
            return end

    # Calculate the current epoch's sparsity based on the pruning schedule
    if not cumulative:
        return start + (end - start) * (epoch / (max_epochs - 1))


gammas = [0.0, 0.25, 0.5, 0.75]

epochs = [i * 2 for i in range(15)]
backbone = [pruning_schedule(0.5, i * 2) for i in range(15)]
gamma_00 = [pruning_schedule(gammas[0], i * 2) for i in range(15)]
gamma_25 = [pruning_schedule(gammas[1], i * 2) for i in range(15)]
gamma_50 = [pruning_schedule(gammas[2], i * 2) for i in range(15)]
gamma_75 = [pruning_schedule(gammas[3], i * 2) for i in range(15)]

# Create a dataframe for plotting
data = {
    "Epochs": epochs,
    "Backbone": backbone,
    "γ = 0.0": gamma_00,
    "γ =0.25": gamma_25,
    "γ =0.5": gamma_50,
    "γ =0.75": gamma_75,
}

df = pd.DataFrame(data)

# Melt the dataframe for seaborn
df_melted = df.melt("Epochs", var_name="Schedule", value_name="Value")

# Plotting
sns.set(style="whitegrid")
plt.figure(figsize=(7, 4.2))

# Plot the Backbone pruning schedule with a different linestyle and marker
sns.lineplot(
    data=df_melted[df_melted["Schedule"] == "Backbone"],
    x="Epochs",
    y="Value",
    hue="Schedule",
    marker="o",
    linestyle="--",
    markersize=8.0,
    linewidth=2.5,
    alpha=0.75,
)

# Plot the Gamma pruning schedules with the same linestyle and marker
sns.lineplot(
    data=df_melted[df_melted["Schedule"] != "Backbone"],
    x="Epochs",
    y="Value",
    hue="Schedule",
    marker="x",
    linestyle="-",
    linewidth=2.5,
    alpha=0.75,
)
plt.subplots_adjust(bottom=0.15)
plt.title("Pruning Rate")
plt.xlabel("Epoch")
plt.ylabel("Sparsity")
plt.legend(title="Pruning Schedule")
plt.savefig("pruning_schedule.jpg", dpi=600)
plt.show()
