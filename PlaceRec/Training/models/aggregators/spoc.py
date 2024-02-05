
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .l2norm import L2Norm


class SPoC(nn.Module):
    def __init__(self, feature_map_shape: torch.tensor, out_dim: int=1024):
        super().__init__()
        self.channel_pool = nn.Conv2d(in_channels=int(feature_map_shape[0]), out_channels=out_dim, kernel_size=1)
        self.pool = nn.AvgPool2d((feature_map_shape[1], feature_map_shape[2]))
        self.norm = L2Norm()

    def forward(self, x):
        B = x.shape[0]
        x = self.channel_pool(x)
        x = self.pool(x).view(B, -1)
        x = self.norm(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + "()"