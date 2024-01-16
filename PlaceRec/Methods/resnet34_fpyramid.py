import os
import pickle
from os.path import join
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm import tqdm
from typing import Union
from ..utils import get_config, s3_bucket_download
from .base_method import BaseModelWrapper



import torch
import torchvision.models as models
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models._utils import IntermediateLayerGetter


def create_feature_pyramid_model(feature_dim=2048):
    # Load a pre-trained ResNet34 model
    resnet34 = models.resnet34(pretrained=True)
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    backbone = IntermediateLayerGetter(resnet34, return_layers=return_layers)
    in_channels_list = [resnet34.layer1[-1].conv2.out_channels, 
                        resnet34.layer2[-1].conv2.out_channels,
                        resnet34.layer3[-1].conv2.out_channels,
                        resnet34.layer4[-1].conv2.out_channels]

    fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=feature_dim//4)
    model = torch.nn.Sequential(backbone, fpn)
    return model


def create_fusion_mlp(n_features=4):
    model = nn.Sequential(
        nn.Linear(2, 12), 
        nn.ReLU(),
        nn.Linear(12, 12), 
        nn.ReLU(), 
        nn.Linear(12, 4),
        nn.Softmax()
    )
    return model


class ResNet34FPyramidModel(nn.Module):
    def __init__(self, feature_dim=2048): 
        super().__init__()
        self.backbone = create_feature_pyramid_model(feature_dim=feature_dim)
        self.fusion_mlp = create_fusion_mlp()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x):
        b, c, h, w = x.shape
        fp = self.backbone(x)
        features1 = self.adaptive_pool(fp['0']).view(b, -1)
        features2 = self.adaptive_pool(fp['1']).view(b, -1)
        features3 = self.adaptive_pool(fp['2']).view(b, -1)
        features4 = self.adaptive_pool(fp['3']).view(b, -1)
        all_features = torch.stack([features1, features2, features3, features4])
        fusion_weights = self.fusion_mlp(torch.tensor([[h/480 - 0.5, w/640 - 0.5]]))
        print(fusion_weights)
        weighted_features = fusion_weights.permute(1, 0)[:, :, None] * all_features
        return weighted_features.sum(0)



        




class ResNet34FPyramid(BaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        model = ResNet34FPyramidModel()
        preprocess = transforms.Compose([
                transforms.Resize((480, 640)),  # Resize the image to the size expected by ResNet34
                transforms.ToTensor(),          # Convert the image to a PyTorch tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using the mean and std used in pre-training
                                    std=[0.229, 0.224, 0.225])
            ])
        
        super().__init__(model=model, preprocess=preprocess, name="resnet34convap")

        # some layers not implemented on metal
        if self.device == "mps":
            self.device = "cpu"
        self.model.to(self.device)

    def set_device(self, device: str) -> None:
        if "mps" in device:
            device = "cpu" 
        self.device = device
        self.model.to(device)



