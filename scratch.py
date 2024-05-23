import torch
import torch.nn as nn
import torchvision.models as models
import torch_pruning as tp
from PlaceRec.Training.GSV_Cities.sparse_utils import pruning_schedule
from PlaceRec.utils import get_method
from parsers import train_arguments


def get_scheduler(args):
    pruning_freq = args.pruning_freq

    def schd(pruning_ratio_dict, steps):
        return [
            pruning_schedule(i * pruning_freq, cumulative=False) * pruning_ratio_dict
            for i in range(steps + 1)
        ]

    return schd


def get_dont_prune(method, args):
    if "mixvpr" in method.name.lower():
        dont_prune = []

        def loop_through(module):
            for name, layer in module.named_children():
                # If a layer has no children, it's a leaf layer
                if len(list(layer.children())) == 0:
                    if name != "row_proj" and isinstance(layer, nn.Linear):
                        dont_prune.append(layer)
                else:
                    loop_through(layer)

        loop_through(method.model)
        return dont_prune
    else:
        return []


def get_pruning_ratio_dict(method, args):
    print(args.aggregation_pruning_rate)
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
        print(layer_dict.keys())
        # Define the pruning ratio dictionary
        pruning_ratio_dict = {
            layer: args.final_sparsity for layer in layer_dict.values()
        }
        for name in layer_dict:
            if "aggregator" in name or "linear" in name:
                pruning_ratio_dict[layer_dict[name]] = args.aggregation_pruning_rate
        return pruning_ratio_dict


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

    pruner = tp.pruner.GroupNormPruner(
        method.model,
        example_img,
        importance,
        iterative_steps=args.max_epochs // args.pruning_freq,
        iterative_pruning_ratio_scheduler=get_scheduler(args),
        pruning_ratio_dict=get_pruning_ratio_dict(method, args),
        ignored_layers=get_dont_prune(method, args),
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

method = get_method("ConvAP", pretrained=False)
layer_names = [name for name, _ in method.model.named_modules()]


method, pruner, orig_nparams = setup_pruner(method, args)


for epoch in range(args.max_epochs // args.pruning_freq):
    if epoch > 0:
        pruner.step()
    sparsity = get_sparsity(method, orig_nparams)
    img = method.example_input().to(next(method.model.parameters()).device)
    out = method.model(img)

    print(f"Epoch: {epoch}  Sparsity: {sparsity:.3f}  Output Dim: {out.shape[1]}")
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
