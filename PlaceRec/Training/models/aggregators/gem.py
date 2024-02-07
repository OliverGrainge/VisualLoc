import torch
import torch.nn.functional as F
import torch.nn as nn
from .l2norm import L2Norm

class GeMPool(nn.Module):
    """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch
    we add flatten and norm so that we can use it as one aggregation layer.
    """
    def __init__(self, feature_map_shape: torch.tensor, out_dim=1024, p=3, eps=1e-6):
        super().__init__()
        self.channel_pool = nn.Conv2d(in_channels=feature_map_shape[0], out_channels=out_dim, kernel_size=1)
        self.p = nn.Parameter(torch.ones(1)*p)
        self.avg_pool = nn.AvgPool2d((feature_map_shape[1], feature_map_shape[2]))
        self.eps = eps
        self.out_dim = out_dim
        self.norm = L2Norm()

    def forward(self, x):
        x = self.channel_pool(x)
        x.clamp(min=self.eps).pow(self.p)
        x = self.avg_pool(x)
        x.pow(1./self.p)
        x = x.flatten(1)
        x = self.norm(x)
        return x
    

class GemPoolTokens(nn.Module):
    def __init__(self, feature_map_shape: torch.tensor, out_dim=1024, p=3, eps=1e-6):
        super().__init__()
        self.token_pool = nn.Linear(feature_map_shape[1], out_dim)
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.norm = L2Norm()
    
    def forward(self, x):
        x = self.token_pool(x)
        x.clamp(min=self.eps).pow(self.p)
        x = torch.mean(x, dim=1)
        x.pow(1./self.p)
        x = self.norm(x)
        return x
