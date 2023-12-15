import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models, transforms
from transformers import ViTFeatureExtractor, ViTModel

from PlaceRec.utils import L2Norm

from .base_method import BaseModelWrapper

filepath = os.path.dirname(os.path.abspath(__file__))


preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),  # This will automatically scale pixels to [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize
    ]
)


class vit_base_patch16_224_gap(nn.Module):
    def __init__(self, feature_dim: int=2048, num_trainable_blocks: int=3):
        super().__init__()
        self.vit_backbone = ViTModel.from_pretrained("google/vit-base-patch16-224")

        # Freeze all transformer blocks except the last `num_trainable_blocks`
        num_blocks = len(self.vit_backbone.encoder.layer)  # Total number of blocks
        for i, block in enumerate(self.vit_backbone.encoder.layer):
            if i < num_blocks - num_trainable_blocks:
                for param in block.parameters():
                    param.requires_grad = False

        self.fc = nn.Linear(768, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit_backbone(x).last_hidden_state
        x = x.mean(1)
        x = self.fc(x)
        return x


class ViT_base_patch16_224_gap(BaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        model = vit_base_patch16_224_gap()
        if pretrained:
            raise Exception("Pre-trained weights are not available")

        super().__init__(model=model, preprocess=preprocess, name="vit_base_patch16_224_gap")

        self.model.to(self.device)
        self.model.eval()
