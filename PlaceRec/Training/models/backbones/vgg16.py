
import torch
import torch.nn as nn
import timm
import numpy as np
import torchvision

def VGG16(model_name='vgg16', pretrained=True, layers_to_freeze=2):
    model = torchvision.models.vgg16(pretrained=True)

    if pretrained:
        for child in model.children():
            for i, param in enumerate(child.parameters()):
                if i >= layers_to_freeze:
                    param.requires_grad = False
        
    return model.features