import torch
import torch.nn.functional as F
import torch.nn as nn
from .l2norm import L2Norm


class ConvAP(nn.Module):
    """Implementation of ConvAP as of https://arxiv.org/pdf/2210.10239.pdf

    Args:
        in_channels (int): number of channels in the input of ConvAP
        out_channels (int, optional): number of channels that ConvAP outputs. Defaults to 512.
        s1 (int, optional): spatial height of the adaptive average pooling. Defaults to 2.
        s2 (int, optional): spatial width of the adaptive average pooling. Defaults to 2.
    """
    def __init__(self, feature_map_shape: torch.tensor, out_dim=1024, s1=2, s2=2):
        super(ConvAP, self).__init__()
        #assert out_dim % 4 == 0
        self.channel_pool = nn.Conv2d(in_channels=feature_map_shape[0], out_channels=out_dim//4, kernel_size=1, bias=True)
        self.AAP = nn.AdaptiveAvgPool2d((s1, s2))
        self.norm = L2Norm()

    def forward(self, x):
        B = x.shape[0]
        x = self.channel_pool(x)
        x = self.AAP(x)
        x = x.view(B, -1)
        x = self.norm(x)
        return x
    

