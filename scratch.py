import torch
import torch.nn as nn
import torchvision.models as models
import torch_pruning as tp
from PlaceRec.Training.GSV_Cities.sparse_utils import pruning_schedule
from PlaceRec.utils import get_method
from parsers import train_arguments
import numpy as np
import PlaceRec.Training.GSV_Cities.utils as utils
from sklearn.cluster import KMeans
import csv


def get_scheduler(args):
    pruning_freq = args.pruning_freq

    def schd(pruning_ratio_dict, steps):
        return [
            pruning_schedule(i * pruning_freq, cumulative=False) * pruning_ratio_dict
            for i in range(steps + 1)
        ]

    return schd


def get_dont_prune(method, args):
    if "netvlad" in method.name.lower():
        dont_prune = []
        for name, module in method.model.named_modules():
            if "channel_pool" in name:
                dont_prune.append(module)
        return dont_prune
    return []


def get_channel_groups(method, args):
    if "vit" in method.name:
        channel_groups = {}
        for m in method.model.modules():
            if isinstance(m, nn.MultiheadAttention):
                channel_groups[m] = m.num_heads
        return channel_groups
    return []


def get_pruning_ratio_dict(method, args):
    print("==============================", args.aggregation_pruning_rate)
    if "convap" in method.name:
        layer_dict = {name: module for name, module in method.model.named_modules()}
        # Define the pruning ratio dictionary
        pruning_ratio_dict = {
            layer: args.final_sparsity for layer in layer_dict.values()
        }
        for name in layer_dict:
            if "aggregator" in name:
                pruning_ratio_dict[layer_dict[name]] = args.aggregation_pruning_rate
        return pruning_ratio_dict

    if "mixvpr" in method.name:
        layer_dict = {name: module for name, module in method.model.named_modules()}
        # Define the pruning ratio dictionary
        pruning_ratio_dict = {
            layer: args.final_sparsity for layer in layer_dict.values()
        }
        for name in layer_dict:
            if "aggregator" in name:
                pruning_ratio_dict[layer_dict[name]] = args.aggregation_pruning_rate
        return pruning_ratio_dict

    if "gem" in method.name:
        layer_dict = {name: module for name, module in method.model.named_modules()}

        # Define the pruning ratio dictionary
        pruning_ratio_dict = {
            layer: args.final_sparsity for layer in layer_dict.values()
        }
        for name in layer_dict:
            if "aggregation" in name or "proj" in name:
                pruning_ratio_dict[layer_dict[name]] = args.aggregation_pruning_rate
        return pruning_ratio_dict

    if "netvlad" in method.name:
        layer_dict = {name: module for name, module in method.model.named_modules()}
        # Define the pruning ratio dictionary
        pruning_ratio_dict = {
            layer: args.final_sparsity for layer in layer_dict.values()
        }
        for name in layer_dict:
            if "aggregator" in name or "linear" in name:
                pruning_ratio_dict[layer_dict[name]] = args.aggregation_pruning_rate
        return pruning_ratio_dict

    if "vit" in method.name:
        layer_dict = {name: module for name, module in method.model.named_modules()}
        pruning_ratio_dict = {
            layer: args.final_sparsity for layer in layer_dict.values()
        }
        for name in layer_dict:
            if "encoder.layers.encoder_layer_11.mlp" in name or "encoder.ln" in name:
                pruning_ratio_dict[layer_dict[name]] = args.aggregation_pruning_rate
        return pruning_ratio_dict


def get_mixvpr_in_channel_proj(method):
    layers = {}
    for name, layer in method.model.named_modules():
        # print(name, layer)
        if name == "aggregator.channel_proj":
            layers["layer"] = layer
        if name == "backbone.model.layer3.5.bn3":
            layers["prev_layer"] = layer
    return layers


def get_mixvpr_out_channel_proj(method):
    layers = {}
    for name, layer in method.model.aggregator.named_modules():
        if name in ["channel_proj", "row_proj"]:
            layers[name] = layer
    return layers


def prune_mixvpr_in_channel_proj(method):
    layers = get_mixvpr_in_channel_proj(method)
    num_prune = layers["layer"].in_features - layers["prev_layer"].num_features
    l1_norm = torch.norm(layers["layer"].weight, p=1, dim=0)
    indices_to_prune = torch.topk(l1_norm, num_prune, largest=False).indices
    all_indices = torch.arange(layers["layer"].in_features)
    indices_to_keep = torch.tensor(
        [idx for idx in all_indices if idx not in indices_to_prune]
    )
    new_weight = layers["layer"].weight[:, indices_to_keep].detach()

    if layers["layer"].bias is not None:
        new_bias = layers[
            "layer"
        ].bias.detach()  # Bias remains unchanged because output features are unchanged
    else:
        new_bias = None
    new_layer = nn.Linear(new_weight.size(1), layers["layer"].out_features)
    new_layer.weight = nn.Parameter(new_weight)
    new_layer.bias = nn.Parameter(new_bias) if new_bias is not None else None
    method.model.aggregator.channel_proj = new_layer


def prune_mixvpr_out_channel_proj(method, step):
    layers = get_mixvpr_out_channel_proj(method)
    amount = pruning_schedule(step, True) * args.aggregation_pruning_rate
    num_out_features = layers["channel_proj"].out_features
    num_prune = int(amount * num_out_features)
    l1_norm = torch.norm(layers["channel_proj"].weight.data, p=1, dim=1)
    indices_to_prune = torch.topk(l1_norm, num_prune, largest=False).indices
    mask = torch.ones(num_out_features, dtype=bool)
    mask[indices_to_prune] = False
    new_weight = layers["channel_proj"].weight.data[mask, :].detach()
    if layers["channel_proj"].bias is not None:
        new_bias = layers["channel_proj"].bias[mask].detach()
    else:
        new_bias = None
    new_layer = nn.Linear(layers["channel_proj"].in_features, new_weight.size(0))
    new_layer.weight = nn.Parameter(new_weight)
    if new_bias is not None:
        new_layer.bias = nn.Parameter(new_bias)
    else:
        new_layer.bias = None
    method.model.aggregator.channel_proj = new_layer


def prune_netvlad_centroids(method, step):
    # Prune centroids
    centroids = method.model.aggregation.centroids
    centroids_data = centroids.detach().cpu().numpy()
    amount = pruning_schedule(step, True) * args.aggregation_pruning_rate
    num_clusters = int((1 - amount) * centroids_data.shape[0])
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(centroids_data)
    clustered_data = kmeans.cluster_centers_
    new_centroids = torch.tensor(clustered_data, dtype=centroids.dtype).to(
        centroids.device
    )
    method.model.aggregation.centroids = torch.nn.Parameter(new_centroids)
    method.model.aggregation.clusters_num = int(new_centroids.shape[0])

    # Calculate L1 norm of each filter in the convolutional layer using torch.norm
    conv_filters = method.model.aggregation.conv.weight
    filter_importance = torch.norm(
        conv_filters, p=1, dim=(1, 2, 3)
    )  # Apply L1 norm across channel, height, and width dimensions

    # Prune convolutional filters
    _, important_indices = torch.topk(
        filter_importance, num_clusters
    )  # Get indices of top filters
    new_conv_weights = conv_filters[important_indices, :, :, :]
    method.model.aggregation.conv.weight = torch.nn.Parameter(new_conv_weights)

    # Optionally, adjust the bias if it exists
    if method.model.aggregation.conv.bias is not None:
        new_conv_biases = method.model.aggregation.conv.bias[important_indices]
        method.model.aggregation.conv.bias = torch.nn.Parameter(new_conv_biases)


def setup_pruner(method, args):
    example_img = method.example_input().to(method.device)
    orig_macs, orig_nparams = tp.utils.count_ops_and_params(method.model, example_img)

    if args.pruning_type is None:
        raise Exception(" For structured pruning, Must choose pruning method.")
    elif args.pruning_type == "magnitude":
        importance = tp.importance.MagnitudeImportance(p=2, group_reduction="mean")
    elif args.pruning_type == "first-order":
        importance = tp.importance.GroupTaylorImportance()
    elif args.pruning_type == "second-order":
        importance = tp.importance.GroupHessianImportance()
    else:
        raise Exception(f"Pruning method {args.pruning_type} is not found")

    if args.method.lower() == "mixvpr":
        pruner = tp.pruner.GroupNormPruner(
            method.model.backbone,
            example_img,
            importance,
            iterative_steps=args.max_epochs // args.pruning_freq,
            iterative_pruning_ratio_scheduler=get_scheduler(args),
            pruning_ratio_dict=get_pruning_ratio_dict(method, args),
            ignored_layers=get_dont_prune(method, args),
            channel_groups=get_channel_groups(method, args),
            global_pruning=False,
        )

        class MixVPRPruner:
            def __init__(self, pruner, method):
                self.pruner = pruner
                self.method = method
                self.epoch = 0

            def step(self):
                self.pruner.step()
                prune_mixvpr_in_channel_proj(self.method)
                prune_mixvpr_out_channel_proj(self.method, self.epoch)
                self.epoch += 1

        pruner = MixVPRPruner(pruner, method)
        return method, pruner, orig_nparams

    elif args.method.lower() == "netvlad":
        pruner = tp.pruner.GroupNormPruner(
            method.model.backbone,
            example_img,
            importance,
            iterative_steps=args.max_epochs // args.pruning_freq,
            iterative_pruning_ratio_scheduler=get_scheduler(args),
            pruning_ratio_dict=get_pruning_ratio_dict(method, args),
            ignored_layers=get_dont_prune(method, args),
            channel_groups=get_channel_groups(method, args),
            global_pruning=False,
        )

        class NetVLADPruner:
            def __init__(self, pruner, method):
                self.pruner = pruner
                self.method = method
                self.epoch = 0

            def step(self):
                self.pruner.step()
                prune_netvlad_centroids(self.method, self.epoch)
                self.epoch += 1

        pruner = NetVLADPruner(pruner, method)

        return method, pruner, orig_nparams

    else:
        pruner = tp.pruner.GroupNormPruner(
            method.model,
            example_img,
            importance,
            iterative_steps=args.max_epochs // args.pruning_freq,
            iterative_pruning_ratio_scheduler=get_scheduler(args),
            pruning_ratio_dict=get_pruning_ratio_dict(method, args),
            ignored_layers=get_dont_prune(method, args),
            channel_groups=get_channel_groups(method, args),
            global_pruning=False,
        )

        return method, pruner, orig_nparams


def get_layer_names(method):
    names = []

    def rec(model):
        for name, child in model.named_children():
            if len(list(child.children())) == 0:
                names.append(name)
            else:
                rec(child)

    rec(method.model)
    return names


def get_sparsity(method, orig_nparams):
    macs, nparams = tp.utils.count_ops_and_params(
        method.model,
        method.example_input().to(next(method.model.parameters()).device),
    )
    return 1 - (nparams / orig_nparams)


args = train_arguments()

method = get_method(args.method, pretrained=False)
method, pruner, orig_nparams = setup_pruner(method, args)
step = 0
# Initialize CSV file and writer
with open(f"{method.name}_latency_measurements.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Epoch",
            "Sparsity",
            "Output_Dim",
            "CPU_Latency_bs1",
            "GPU_Latency_bs1",
            "CPU_Latency_bs50",
            "GPU_Latency_bs50",
        ]
    )

    step = 0
    for epoch in range(args.max_epochs // args.pruning_freq):
        if epoch > 0:
            pruner.step()

        sparsity = get_sparsity(method, orig_nparams)
        img = method.example_input().to(next(method.model.parameters()).device)
        out = method.model(img)
        cpu_lat1 = utils.measure_cpu_latency(
            method.model, method.example_input(), batch_size=1
        )
        gpu_lat1 = utils.measure_gpu_latency(
            method.model, method.example_input(), batch_size=1
        )
        cpu_lat50 = utils.measure_cpu_latency(
            method.model, method.example_input(), batch_size=50
        )
        gpu_lat50 = utils.measure_gpu_latency(
            method.model, method.example_input(), batch_size=50
        )

        # Append a row for each epoch
        writer.writerow(
            [
                epoch,
                sparsity,
                int(out.shape[1]),
                cpu_lat1,
                gpu_lat1,
                cpu_lat50,
                gpu_lat50,
            ]
        )

        step += 1
        print(f"Epoch: {epoch}  Sparsity: {sparsity:.3f}  Output Dim: {out.shape}")

"""
import numpy as np 
import matplotlib.pyplot as plt

method = get_method("ResNet18_ConvAP", pretrained=False)
method2 = get_method("ResNet18_MixVPR", pretrained=False)

def get_layer_params(model):
    layer_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_params.append((name, param.numel()))
    return layer_params

# Assuming method and method2 are already defined and contain the models
# method = get_method("ResNet18_ConvAP", pretrained=False)
# method2 = get_method("ResNet18_MixVPR", pretrained=False)
convap = method.model
mixvpr = method2.model

# Get layer parameters for both models
convap_layer_params = get_layer_params(convap)
mixvpr_layer_params = get_layer_params(mixvpr)

# Function to plot the layer-wise parameters side by side
def plot_layer_params_side_by_side(layer_params1, layer_params2, title1, title2):
    layers1, params1 = zip(*layer_params1)
    layers2, params2 = zip(*layer_params2)
    
    indices1 = np.arange(len(layers1))
    indices2 = np.arange(len(layers2))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    axes[0].bar(indices1, params1, color='skyblue')
    axes[0].set_xlabel('Layers')
    axes[0].set_ylabel('Number of Parameters')
    axes[0].set_title(title1)
    axes[0].set_xticks(indices1)
    axes[0].set_xticklabels(layers1, rotation=90)

    axes[1].bar(indices2, params2, color='lightgreen')
    axes[1].set_xlabel('Layers')
    axes[1].set_ylabel('Number of Parameters')
    axes[1].set_title(title2)
    axes[1].set_xticks(indices2)
    axes[1].set_xticklabels(layers2, rotation=90)

    plt.tight_layout()
    plt.show()

# Plot the layer-wise parameters side by side
plot_layer_params_side_by_side(convap_layer_params, mixvpr_layer_params, 'Layer-wise Parameters for ResNet18_ConvAP', 'Layer-wise Parameters for ResNet18_MixVPR')

"""
