import torch
import torch.nn as nn
from transformers import ViTModel
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from torchvision import transforms
from PlaceRec.Methods import BaseModelWrapper


preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),  # This will automatically scale pixels to [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize
    ]
)

class NetVLAD(nn.Module):
    def __init__(self, num_clusters, dim):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.clusters = nn.Parameter(torch.randn(num_clusters, dim))
        self.clusters2 = nn.Parameter(torch.randn(1, num_clusters, dim))

    def forward(self, x):
        x = x.unsqueeze(-1)  # Add a dimension for clusters
        assignment = torch.softmax(torch.sum(x * self.clusters, dim=2), dim=2)
        a_sum = assignment.sum(-2).unsqueeze(-2)
        a = a_sum * self.clusters2

        vlad = torch.sum(assignment * x - a, dim=1)
        return vlad

class ViTNetVLAD(nn.Module):
    def __init__(self, num_clusters):
        super(ViTNetVLAD, self).__init__()
        self.vit_backbone = ViTModel.from_pretrained("google/vit-base-patch16-224")
        dim = self.vit_backbone.config.hidden_size  # Dimension of the token embeddings from ViT
        self.netvlad = NetVLAD(num_clusters, dim)

    def forward(self, x):
        outputs = self.vit_backbone(x)
        tokens = outputs.last_hidden_state
        vlad = self.netvlad(tokens)
        return vlad


class ViT_base_patch16_224_gap(BaseModelWrapper):
    def __init__(self, pretrained: bool = True, num_clusters: int=24):
        model = ViTNetVLAD(24)
        if pretrained:
            raise Exception("Pre-trained weights are not available")
        super().__init__(model=model, preprocess=preprocess, name="vit_base_patch16_224_netvlad")
        self.model.to(self.device)
        self.model.eval()
