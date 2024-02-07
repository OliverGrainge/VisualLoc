import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .l2norm import L2Norm


class MixVPR(nn.Module):
    def __init__(self, feature_map_shape, out_dim=1024): 
        super().__init__()
        self.mix1 = nn.Sequential(
            nn.LayerNorm(feature_map_shape[1] * feature_map_shape[2]),
            nn.Linear(feature_map_shape[1] * feature_map_shape[2], feature_map_shape[1] * feature_map_shape[2]),
            nn.ReLU(),
            nn.Linear(feature_map_shape[1] * feature_map_shape[2], feature_map_shape[1] * feature_map_shape[2]),
        )

        self.mix2 = nn.Sequential(
            nn.LayerNorm(feature_map_shape[1] * feature_map_shape[2]),
            nn.Linear(feature_map_shape[1] * feature_map_shape[2], feature_map_shape[1] * feature_map_shape[2]),
            nn.ReLU(),
            nn.Linear(feature_map_shape[1] * feature_map_shape[2], feature_map_shape[1] * feature_map_shape[2]),
        )

        self.mix3 = nn.Sequential(
            nn.LayerNorm(feature_map_shape[1] * feature_map_shape[2]),
            nn.Linear(feature_map_shape[1] * feature_map_shape[2], feature_map_shape[1] * feature_map_shape[2]),
            nn.ReLU(),
            nn.Linear(feature_map_shape[1] * feature_map_shape[2], feature_map_shape[1] * feature_map_shape[2]),
        )

        self.mix4 = nn.Sequential(
            nn.LayerNorm(feature_map_shape[1] * feature_map_shape[2]),
            nn.Linear(feature_map_shape[1] * feature_map_shape[2], feature_map_shape[1] * feature_map_shape[2]),
            nn.ReLU(),
            nn.Linear(feature_map_shape[1] * feature_map_shape[2], feature_map_shape[1] * feature_map_shape[2]),
        )


        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.channel_proj = nn.Linear(feature_map_shape[0], out_dim // 4)
        self.row_proj = nn.Linear(feature_map_shape[1] * feature_map_shape[2], 4)
        self.norm = L2Norm()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)
        x = self.mix1(x) + x
        x = self.mix2(x) + x
        x = self.mix3(x) + x
        x = self.mix4(x) + x
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = x.view(B, -1)
        x = self.norm(x)
        return x




class MixVPRTokens(nn.Module):
    def __init__(self, feature_map_shape: torch.Tensor, out_dim: int=1024):
        super().__init__()
        dim = feature_map_shape[1]

        self.mix1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.mix2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.mix3 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.mix4 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        self.channel_proj = nn.Linear(feature_map_shape[0], out_dim // 4)
        self.row_proj = nn.Linear(feature_map_shape[1], 4)
        self.norm = L2Norm()

    def forward(self, x):
        B, N, D = x.shape
        x = self.mix1(x) + x
        x = self.mix2(x) + x
        x = self.mix3(x) + x 
        x = self.mix4(x) + x # dim 
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = x.view(B, -1)
        x = self.norm(x)
        return x
        

