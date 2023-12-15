import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models, transforms
from transformers import AutoModel

from PlaceRec.utils import L2Norm

from .base_method import BaseModelWrapper

filepath = os.path.dirname(os.path.abspath(__file__))

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.ToTensor(),  # This replaces the manual rescaling
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class dinov2_base_patch16_224_gap(nn.Module):
    def __init__(self, feature_dim=2048, num_trainable_blocks=3):
        super().__init__()
        self.vit_backbone = AutoModel.from_pretrained("facebook/dinov2-base")

        # Freeze all transformer blocks except the last `num_trainable_blocks`
        num_blocks = len(self.vit_backbone.encoder.layer)  # Total number of blocks
        for i, block in enumerate(self.vit_backbone.encoder.layer):
            if i < num_blocks - num_trainable_blocks:
                for param in block.parameters():
                    param.requires_grad = False

        self.fc = nn.Linear(768, feature_dim)

    def forward(self, x):
        x = self.vit_backbone(x).last_hidden_state
        x = x.mean(1)
        x = self.fc(x)
        return x


class DinoV2_base_patch16_224_gap(BaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        model = dinov2_base_patch16_224_gap()
        if pretrained:
            raise Exception("Pre-trained weights are not available")

        super().__init__(model=model, preprocess=preprocess, name="dinov2_base_patch16_224_gap")

        self.model.to(self.device)
        self.model.eval()
