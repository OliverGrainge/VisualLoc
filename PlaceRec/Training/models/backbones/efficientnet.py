import torch.nn as nn
from efficientnet_pytorch import EfficientNet



import torch
import torch.nn as nn
import timm
import numpy as np

def EfficientNet(model_name='efficientnet_b0', pretrained=True, layers_to_freeze=2):
    model = timm.create_model(model_name=model_name, pretrained=pretrained)

    if pretrained:
        if layers_to_freeze >= 0:
            model.conv_stem.requires_grad_(False)
            model.blocks[0].requires_grad_(False)
            model.blocks[1].requires_grad_(False)
        if layers_to_freeze >= 1:
            model.blocks[2].requires_grad_(False)
        if layers_to_freeze >= 2:
            model.blocks[3].requires_grad_(False)
        if layers_to_freeze >= 3:
            model.blocks[4].requires_grad_(False)
        if layers_to_freeze >= 4:
            model.blocks[5].requires_grad_(False)
        
    backbone = nn.Sequential(*list(model.children())[:-2])
    return backbone

