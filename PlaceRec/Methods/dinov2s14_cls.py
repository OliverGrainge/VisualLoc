import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import PlaceRec.utils as utils
from PlaceRec.Methods import SingleStageBaseModelWrapper

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize
    ]
)


class VitWrapper(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit_model = vit_model
        self._freeze_up_to_last_transformer_block()

    def _freeze_up_to_last_transformer_block(self):
        # Assuming 'self.vit_model' has an attribute 'blocks' that contains the transformer blocks
        # Freeze all transformer blocks except the last one
        for param in self.vit_model.patch_embed.parameters():
            param.requires_grad = False

        for block in self.vit_model.blocks[:-2]:
            for param in block.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.vit_model(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x


class DINOv2S14_CLS(SingleStageBaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        model = VitWrapper(torch.hub.load("facebookresearch/dinov2", "dinov2_vits14"))
        sys.stdout = original_stdout

        super().__init__(model=model, preprocess=preprocess, name="dinov2s14_cls")
