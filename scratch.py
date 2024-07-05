import torch
import torch.nn as nn

import torch
import torch.nn as nn
from PlaceRec.Training.GSV_Cities.sparse_utils import pruning_schedule
from parsers import train_arguments
from PlaceRec.utils import get_method


method = get_method("ResNet34_ConvAP", False)
print(method.model)
"""
args = train_arguments()
args.max_epochs = 30
args.pruning_freq = 3
args.aggregation_pruning_rate = 0.5


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(10, 100, 1)
    
    def forward(self, x):
        x = self.conv(x)
        return x.flatten(1)
    

def prune_conv_layer_by_l2(in_channels: int, layer: nn.Conv2d, epoch, args):

    # Check if the layer is an instance of nn.Conv2d
    if not isinstance(layer, nn.Conv2d):
        raise ValueError("The layer to be pruned must be an instance of nn.Conv2d")

    # Calculate the pruning amount for output channels
    amount = pruning_schedule(args, epoch * args.pruning_freq, True)

    if in_channels < layer.in_channels:
        # Compute L2 norm across the spatial dimensions of the kernel weights
        l2_norms_in = torch.norm(layer.weight.data.view(layer.out_channels, layer.in_channels, -1), dim=2, p=2).mean(dim=0)
        num_channels_to_prune_in = layer.in_channels - in_channels
        prune_indices_in = torch.argsort(l2_norms_in)[:num_channels_to_prune_in]
        keep_indices_in = torch.argsort(l2_norms_in)[num_channels_to_prune_in:]

        # Update weights for input channel pruning
        layer.weight.data = layer.weight.data[:, keep_indices_in, :, :].clone()

        # Update the in_channels attribute
        layer.in_channels = in_channels

    if amount == 0:
        return

    # Prune output channels
    l2_norms_out = torch.norm(layer.weight.data.view(layer.out_channels, -1), dim=1, p=2)
    num_out_channels = layer.out_channels
    #print(num_out_channels)
    #num_channels_to_prune_out = int(amount * num_out_channels)
    #print(amount)

    num_channels_to_prune_out = int(amount * 100)
    prune_indices_out = torch.argsort(l2_norms_out)[:num_channels_to_prune_out]
    keep_indices_out = torch.argsort(l2_norms_out)[num_channels_to_prune_out:]

    # Update weights and biases for output channel pruning
    layer.weight.data = layer.weight.data[keep_indices_out, :, :, :].clone()
    if layer.bias is not None:
        layer.bias.data = layer.bias.data[keep_indices_out].clone()

    # Update the out_channels attribute
    layer.out_channels -= num_channels_to_prune_out

x = torch.randn(1, 10, 5, 5)
model = Model()
out = model(x)

amounts = []
total = 0
for i in range(args.max_epochs//args.pruning_freq + 1):
    amount = pruning_schedule(args, i*args.pruning_freq, True)
    total += amount
    #print("pruning step", i, "pruning amount", amount, "pruning_total", total)
    prune_conv_layer_by_l2(10, model.conv, i, args)
    out = model(x)
    print("Step: ", i, "Sparsity: ", model.conv.out_channels/100)
"""
