import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .l2norm import L2Norm

class MAC(nn.Module):
    def __init__(self, feature_map_shape: torch.tensor, out_dim: int=1024):
        super().__init__()
        self.channel_pool = nn.Conv2d(in_channels=feature_map_shape[0], out_channels=out_dim, kernel_size=1)
        self.pool = nn.MaxPool2d((feature_map_shape[1], feature_map_shape[2]))
        self.norm = L2Norm()

    def forward(self, x):
        B = x.shape[0]
        x = self.channel_pool(x)
        x = self.pool(x).view(B, -1)
        x = self.norm(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + "()"
    

class MACTokens(nn.Module):
    def __init__(self, feature_map_shape: torch.tensor, out_dim: int=1024):
        super().__init__()
        self.token_pool = nn.Linear(feature_map_shape[1], out_dim)
        self.norm = L2Norm()

    def forward(self, x):
        x = self.token_pool(x)
        x, _ = torch.max(x, dim=1)
        x = self.norm(x)
        return x
