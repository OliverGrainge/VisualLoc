import torch
import torchvision.models as models
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models._utils import IntermediateLayerGetter
from PIL import Image
import numpy as np
from torchvision import transforms
from torch import nn


import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, x, weights):
        branch1x1 = self.branch1x1(x) * weights[:, 0][:, None, None, None]
        branch3x3 = self.branch3x3(x) * weights[:, 1][:, None, None, None]
        branch5x5 = self.branch5x5(x) * weights[:, 2][:, None, None, None]
        print(weights.shape, branch1x1.shape, branch3x3.shape, branch5x5.shape)

        fused_outputs = torch.stack([branch1x1, branch3x3, branch5x5]).mean(0)
        return fused_outputs
    

class GatingFunction(nn.Module):
    def __init__(self, in_channels, n_weights):
        super(GatingFunction, self).__init__()
        self.conv = nn.Conv2d(in_channels, n_weights, kernel_size=1)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.softmax = nn.Softmax()
        self.n_weights = n_weights

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv(x)
        x = self.pool(x).view(b, self.n_weights)
        x = self.softmax(x)
        return x
    

class GatingInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatingInceptionBlock, self).__init__()
        self.inception_module = InceptionBlock(in_channels=in_channels, 
                                               out_channels=out_channels)
        self.gating_module = GatingFunction(in_channels=in_channels,
                                            n_weights=3)
    
    def forward(self, x):
        fusion_weights = self.gating_module(x)
        fused_features = self.inception_module(x, fusion_weights)
        return fused_features



input = torch.randn(10, 128, 32, 32)
block = GatingInceptionBlock(128, 256)
out = block(input)
print(out.shape)

    
"""

class GatingFunction(nn.Module):
    def __init__(self, in_channels, gating_channels):
        super(GatingFunction, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, gating_channels, kernel_size=1)
        self.fc = nn.Linear(gating_channels, gating_channels)

    def forward(self, x):
        x = self.conv1x1(x)
        n, c, h, w = x.size()
        x = x.view(n, c, h * w)
        x = x.permute(0, 2, 1).contiguous()  # Change to (n, h*w, c)
        x = x.view(n * h * w, c)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        x = x.view(n, h, w, c)
        x = x.permute(0, 3, 1, 2).contiguous()  # Change back to (n, c, h, w)
        return x
    


    


class InceptionBlockWithGating(nn.Module):
    def __init__(self, in_channels, gating_channels=32):
        super(InceptionBlockWithGating, self).__init__()
        # Inception branches
        self.branch1x1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch3x3 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.branch5x5 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)

        # Gating functions for each branch
        self.gate1x1 = GatingFunction(64, gating_channels)
        self.gate3x3 = GatingFunction(128, gating_channels)
        self.gate5x5 = GatingFunction(32, gating_channels)

    def forward(self, x):
        # Apply inception branches
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)

        # Apply gating
        gate1x1 = self.gate1x1(branch1x1)
        gate3x3 = self.gate3x3(branch3x3)
        gate5x5 = self.gate5x5(branch5x5)

        # Element-wise multiplication (Hadamard product)
        gated_branch1x1 = branch1x1 * gate1x1
        gated_branch3x3 = branch3x3 * gate3x3
        gated_branch5x5 = branch5x5 * gate5x5

        # Concatenate the gated branches
        outputs = [gated_branch1x1, gated_branch3x3, gated_branch5x5]
        return torch.cat(outputs, 1)
    

block = InceptionBlock(128)
"""