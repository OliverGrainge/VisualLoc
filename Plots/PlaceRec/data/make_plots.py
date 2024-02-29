import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv("results.csv")



backbones = {"dinov2":"DINOv2",
         "efficientnet": "EfficientNet", 
         "resnet101": "ResNet101",
         "resnet50": "ResNet50",
         "resnet18": "ResNet18",
         "mobilenet": "MobileNetV2",
         "squeezenet": "SqueezeNet", 
         "vgg16":"VGG16"}

aggregations = {"gem": "GeM",
                "spoc": "SPoC",
                "netvlad": "NetVLAD",
                "mac": "MAC",
                "mixvpr": "MixVPR"}

precisions = {"fp32":"fp32",
              "fp16":"fp16",
              "int8":"int8",
              }

aggregation_order = ["SPoC", "MAC", "GeM", "NetVLAD", "MixVPR"]
desired_column_order = ["ResNet18", "ResNet50", "ResNet101", "MobileNetV2", "EfficientNet", "SqueezeNet", "VGG16", "DINOv2"]

data[['backbone', 'aggregation', 'descriptor_size', 'precision']] = data['id'].str.split('_', expand=True)
data.set_index("id", inplace=True)

data = data.replace(aggregations)
data = data.replace(backbones)
data.replace(precisions)



updated_retrieval_metrics = ["pitts30k_recall@1", "mapillarysls_recall@1", "nordland_recall@1"]
fm = ["Pitts30k R@1", "MSLS R@1", "Nordland R@1"]
fig, ax = plt.subplots(len(updated_retrieval_metrics), 1, figsize=(21, 20))

for i, metric in enumerate(updated_retrieval_metrics):
    sns.boxplot(
        data=data, x="backbone", y=metric, hue="precision", ax=ax[i], palette="Set2", hue_order=["fp32", "fp16", "int8"], order=desired_column_order, showfliers=False, linewidth=2.7
    )
    ax[i].set_title(
        f"Distribution of {fm[i]} by Backbone and Precision", fontsize="37"
    )
    ax[i].set_ylabel(f"{fm[i]} Score", fontsize="31")
    ax[i].set_xlabel("Backbone", fontsize="32")
    ax[i].set_xticklabels(ax[i].get_xticklabels(), fontsize=25)
    ax[i].tick_params(axis='y', labelsize=21)
    if i == 2:
        ax[i].legend(title="Precision", loc="upper right", fontsize=22)
    else:
        ax[i].legend(title="Precision", loc="lower left", fontsize=22)
plt.subplots_adjust(hspace=0.38, top=0.94, bottom=0.07, left=0.08, right=0.95)
plt.savefig("backbone_performance.png", dpi=500)


updated_retrieval_metrics = ["pitts30k_recall@1", "mapillarysls_recall@1", "nordland_recall@1"]
fig, ax = plt.subplots(len(updated_retrieval_metrics), 1, figsize=(22, 20))

for i, metric in enumerate(updated_retrieval_metrics):
    sns.boxplot(
        data=data, x="aggregation", y=metric, hue="precision", ax=ax[i], palette="pastel", hue_order=["fp32", "fp16", "int8"], showfliers=False, order=aggregation_order, linewidth=2.7
    )
    ax[i].set_title(
        f"Distribution of {fm[i]} by Pooling Method and Precision", fontsize="31"
    )
    ax[i].set_ylabel(f"{fm[i]} Score", fontsize="29")
    ax[i].set_xlabel("Pooling", fontsize="29")
    ax[i].set_xticklabels(ax[i].get_xticklabels(), fontsize=29)
    ax[i].tick_params(axis='y', labelsize=18)
    if i == 2:
        ax[i].legend(title="Precision", loc="upper right", fontsize=21)
    else:
        ax[i].legend(title="Precision", loc="lower right", fontsize=21)
plt.subplots_adjust(hspace=0.29, top=0.96, bottom=0.06, left=0.09, right=0.96)

plt.savefig("aggregation_recall.png", dpi=500)





# Plotting the effect of backbone on encoding latency and memory size
fig, ax = plt.subplots(2, 1, figsize=(16, 12))
# Encoding Latency
sns.boxplot(
    data=data,
    x="backbone",
    y="gpu_embedded_latency",
    hue="precision",
    ax=ax[0],
    palette="Set2",
    hue_order=["fp32", "fp16", "int8"],
    order=desired_column_order,
    showfliers=False,
    log_scale=True,
    linewidth=2.7
)
ax[0].set_title("Feature Encoding Time by Backbone and Precision", fontsize="26")
ax[0].set_ylabel("Encoding Time (ms)", fontsize="23")
ax[0].set_xlabel("Backbone", fontsize="23")
#ax[0].set_ylim(0, 75)
ax[0].set_xticklabels(ax[0].get_xticklabels(), fontsize=19)
ax[0].legend(title="Precision", loc="lower right", fontsize=19)
# Memory Size
sns.boxplot(
    data=data, x="backbone", y="memory", hue="precision", hue_order=["fp32", "fp16", "int8"], ax=ax[1], palette="Set2", order=desired_column_order,showfliers=False, linewidth=2.7
)
ax[1].set_title("Distribution of Model Size by Backbone and Precision", fontsize="26")
ax[1].set_ylabel("Model Size (MB)", fontsize="23")
ax[1].set_xlabel("Backbone", fontsize="23")
ax[1].set_xticklabels(ax[1].get_xticklabels(), fontsize=19)
ax[1].legend(title="Precision", loc="upper right", fontsize=19)
ax[0].tick_params(axis='y', labelsize=16)
ax[1].tick_params(axis='y', labelsize=16)
plt.subplots_adjust(hspace=0.29, top=0.96, bottom=0.06, left=0.07, right=0.95)
plt.savefig("Backbone_resource_dist.png", dpi=500)



fig, ax = plt.subplots(2, 1, figsize=(18, 12))

# Encoding Latency
sns.boxplot(
    data=data,
    x="aggregation",
    y="gpu_embedded_latency",
    hue="precision",
    ax=ax[0],
    palette="pastel",
    #palette="Set2",
    hue_order=["fp32", "fp16", "int8"],
    order=aggregation_order,
    showfliers=False,
    linewidth=1.8
)
ax[0].set_title("Feature Encoding Time by Pooling Method and Precision", fontsize="25")
ax[0].set_ylabel("Encoding Time (ms)", fontsize="19")
ax[0].set_xlabel("Pooling", fontsize="21")
ax[0].set_xticklabels(ax[0].get_xticklabels(), fontsize=21)
ax[0].legend(title="Precision", loc="upper right", fontsize=16)
ax[0].tick_params(axis='y', labelsize=17)
# Memory Size
sns.boxplot(
    data=data, x="aggregation", y="memory", palette="pastel", hue="precision", hue_order=["fp32", "fp16", "int8"], ax=ax[1], order=aggregation_order,showfliers=False, linewidth=1.8
)
ax[1].set_title("Distribution of Model Size by Pooling Method and Precision", fontsize="25")
ax[1].set_ylabel("Model Size (MB)", fontsize="23")
ax[1].set_xlabel("Pooling", fontsize="21")
ax[1].set_xticklabels(ax[1].get_xticklabels(), fontsize=21)
ax[1].legend(title="Precision", loc="upper right", fontsize=19)
ax[1].tick_params(axis='y', labelsize=17)

plt.subplots_adjust(hspace=0.29, top=0.96, bottom=0.06, left=0.09, right=0.95)
plt.savefig("aggregation_memory.png", dpi=500)



# Plotting retrieval performance for updated metrics across aggregation methods
fig, ax = plt.subplots(len(updated_retrieval_metrics), 1, figsize=(15, 20))


for i, metric in enumerate(updated_retrieval_metrics):
    sns.boxplot(
        data=data, x="aggregation", y=metric, hue="precision", ax=ax[i], palette="Set3", hue_order=["fp32", "fp16", "int8"], order=aggregation_order, linewidth=2.7
    )
    ax[i].set_title(f"Distribution of {metric} by Aggregation and Precision")
    ax[i].set_ylabel(f"{metric} Score")
    ax[i].set_xlabel("Aggregation Method")
    ax[i].legend(title="Precision", loc="center right")


# Scatter plot of mean_encoding_time vs. model_size colored by backbone
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=data,
    x="gpu_embedded_latency",
    y="memory",
    hue="backbone",
    palette="deep",
    s=100,
    edgecolor="w",
    alpha=0.7,
)
plt.title("Mean Encoding Time vs. Model Size by Backbone")
plt.xlabel("Mean Encoding Time (ms)")
plt.ylabel("Model Size (bytes)")
plt.legend(title="Backbone", loc="upper right")

plt.savefig("backbone_resource_dist2.png", dpi=500)



# Box plot of model size across aggregation methods
plt.figure(figsize=(9, 5))



sns.boxplot(
    data=data, x="aggregation", y="memory", hue="precision", palette="muted", hue_order=["fp32", "fp16", "int8"], order=aggregation_order, linewidth=2.7
)



plt.title("Distribution of Model Size by Aggregation and Precision")
plt.xlabel("Aggregation Method")
plt.ylabel("Model Size (bytes)")
plt.legend(title="Precision", loc="upper right")


# Box plot of mean encoding time across aggregation methods
plt.figure(figsize=(12, 8))

sns.boxplot(
    data=data,
    x="aggregation",
    y="gpu_embedded_latency",
    hue="precision",
    palette="pastel",
    hue_order=["fp32", "fp16", "int8"],
    order=aggregation_order,
    showfliers=False,
    linewidth=2.7
)

plt.title("Feature Encoding Time by Aggregation and Precision", fontsize="16")
plt.xlabel("Aggregation Method", fontsize="14")
plt.ylabel("Encoding Time (ms)", fontsize="14")
plt.legend(title="Precision", loc="upper right")

plt.savefig("aggregation_memory2.png", dpi=600)





heatmap_data = data.pivot_table(
    values="pitts30k_recall@1", index="aggregation", columns="backbone", aggfunc="mean"
)
# Plot the heatmap with the 'coolwarm' color palette
plt.figure(figsize=(10, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    cmap="coolwarm",
    cbar_kws={"label": "Average pitts30k_r@1 Score"},
)
plt.title("Average Pitts30k R@1 Score by Backbone and Aggregation")
plt.xlabel("Backbone")
plt.ylabel("Aggregation Method")

# plt.show()


data = data[data["precision"] == "int8"]
data.rename(columns={'aggregation': 'Pooling'}, inplace=True)
aggregation_order.reverse()

# Pivot the data to get the mean scores for each metric based on backbone and aggregation
pitts30k_data = data.pivot_table(
    values="pitts30k_recall@1", index="Pooling", columns="backbone", aggfunc="max"
)
pitts30k_data = pitts30k_data.reindex(aggregation_order)
pitts30k_data = pitts30k_data[desired_column_order]
#pitts30k_data = pitts30k_data.rename(columns={"backbone", "Backbone"})

nordland_data = data.pivot_table(
    values="nordland_recall@1", index="Pooling", columns="backbone", aggfunc="max"
)
nordland_data = nordland_data.reindex(aggregation_order)
nordland_data = nordland_data[desired_column_order]
st_lucia_data = data.pivot_table(
    values="mapillarysls_recall@1", index="Pooling", columns="backbone", aggfunc="max"
)
st_lucia_data = st_lucia_data.reindex(aggregation_order)
st_lucia_data = st_lucia_data[desired_column_order]

# Combine the heatmaps into a single figure

fig, ax = plt.subplots(1, 3, figsize=(20, 6))

# Plot the heatmaps
sns.heatmap(
    pitts30k_data,
    annot=True,
    cmap="coolwarm",
    cbar_kws={"label": "Pitts30k R@1 Score"},
    ax=ax[0],
)
cbar = ax[0].collections[0].colorbar
cbar.set_label('Pitts30k R@1 Score', size=14)  

sns.heatmap(
    nordland_data,
    annot=True,
    cmap="coolwarm",
    cbar_kws={"label": "Nordland R@1 Score"},
    ax=ax[2],
) 
cbar = ax[2].collections[0].colorbar
cbar.set_label('Nordland R@1 Score', size=14)  
sns.heatmap(
    st_lucia_data,
    annot=True,
    cmap="coolwarm",
    cbar_kws={"label": "MSLS R@1 Score"},
    ax=ax[1],
)
cbar = ax[1].collections[0].colorbar
cbar.set_label('MSLS r@1 Score', size=14)  
# Set titles
ax[0].set_title("Pitts30k R@1 Score by Backbone and Pooling", fontsize=15)
ax[2].set_title("Nordland R@1 Score by Backbone and Pooling", fontsize=15)
ax[1].set_title("MSLS R@1 Score by Backbone and Pooling", fontsize=15)

for a in ax.flat:
    plt.sca(a)
    plt.xticks(rotation=45, fontsize=14)
    plt.xlabel("Backbone", fontsize=14)
    plt.ylabel("Pooling", fontsize=13)
    plt.yticks(fontsize=13)


plt.tight_layout()
plt.savefig("all_recalls.png", dpi=500)
# plt.show()
"""

filtered_data = data[data["backbone"] == "mobilenetv2conv4"]
# Filter the data further to only include rows with the "GEM" aggregation type
gem_filtered_data = filtered_data[filtered_data["aggregation"] == "gem"]
# print(gem_filtered_data.keys())
# gem_filtered_data["backbone", "aggregation", "descriptor_size", "precision"]


# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(
    gem_filtered_data["descriptor_size"],
    gem_filtered_data["pitts30k_r@1"],
    color="blue",
    alpha=0.7,
)
plt.title("Influence of Descriptor Size on Pitts30k R@1 Performance (GEM Aggregation)")
plt.xlabel("Descriptor Size")
plt.ylabel("Pitts30k R@1 (%)")
plt.grid(True)
plt.tight_layout()
"""
"""
backbone = "mobilenetv2conv4"
aggregation = "mac"
df = pd.read_csv("results.csv")

dataset = "pitts30k_r@1"
df = df[df["backbone"] == backbone]
df = df[df["aggregation"] == aggregation]

df = df[["precision", "fc_output_dim", "pitts30k_r@1", "nordland_r@1", "st_lucia_r@1"]]



fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
precision_mapping = {"fp32_comp": 1, "fp16_comp": 2, "int8_comp": 3}
df["precision_numeric"] = df["precision"].map(precision_mapping)
# Scatter plot
sc = ax.scatter(
    df["precision_numeric"],
    df["fc_output_dim"],
    df[dataset],
    c=df[dataset],
    cmap="viridis",
    s=60,
)

# Surface plot
ax.plot_trisurf(
    df["precision_numeric"], df["fc_output_dim"], df[dataset], cmap="viridis", alpha=0.5
)




ax.set_xlabel("Precision")
ax.set_ylabel("fc_output_dim")
ax.set_zlabel(dataset)
ax.set_xticks(list(precision_mapping.values()))
ax.set_xticklabels(list(precision_mapping.keys()))
ax.view_init(elev=80, azim=0)

# Colorbar
cbar = fig.colorbar(sc)
cbar.ax.set_ylabel("pitts30k_r@1 values")

plt.title("3D Surface Plot of precision, fc_output_dim against pitts30k_r@1")
plt.show()
"""