import os
from os.path import join

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from PlaceRec.utils import L2Norm, get_config

from .base_method import BaseModelWrapper

filepath = os.path.dirname(os.path.abspath(__file__))
config = get_config()


class GeM(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1)))
            .pow(1.0 / self.p)
            .view(x.shape[0], -1)
        )


class Resnet50gemModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        layers = list(backbone.children())[:-3]
        self.backbone = torch.nn.Sequential(*layers)
        self.aggregation = nn.Sequential(L2Norm(), GeM(), nn.Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((480, 640), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ResNet50_GeM(BaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        model = Resnet50gemModel()
        name = "resnet50_gem"
        weight_path = join(config["weights_directory"], name + ".ckpt")
        if pretrained:
            if not os.path.exists(weight_path):
                raise Exception(f"Could not find weights at {weight_path}")
            model.load_state_dict(torch.load(weight_path))

        super().__init__(model=model, preprocess=preprocess, name=name)
