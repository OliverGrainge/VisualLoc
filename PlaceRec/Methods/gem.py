import os
from os.path import join

import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torchvision import transforms

from PlaceRec.utils import L2Norm, get_config

from .base_method import SingleStageBaseModelWrapper

filepath = os.path.dirname(os.path.abspath(__file__))
config = get_config()


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (10, 10))  # Use the fixed size
        x = x.pow(1.0 / self.p)
        return x.view(x.shape[0], -1)


class ResNet(nn.Module):
    def __init__(
        self,
        model_name="resnet50",
        pretrained=True,
        layers_to_freeze=2,
        layers_to_crop=[],
    ):
        """Class representing the resnet backbone used in the pipeline
        we consider resnet network as a list of 5 blocks (from 0 to 4),
        layer 0 is the first conv+bn and the other layers (1 to 4) are the rest of the residual blocks
        we don't take into account the global pooling and the last fc

        Args:
            model_name (str, optional): The architecture of the resnet backbone to instanciate. Defaults to 'resnet50'.
            pretrained (bool, optional): Whether pretrained or not. Defaults to True.
            layers_to_freeze (int, optional): The number of residual blocks to freeze (starting from 0) . Defaults to 2.
            layers_to_crop (list, optional): Which residual layers to crop, for example [3,4] will crop the third and fourth res blocks. Defaults to [].

        Raises:
            NotImplementedError: if the model_name corresponds to an unknown architecture.
        """
        super().__init__()
        self.model_name = model_name.lower()
        self.layers_to_freeze = layers_to_freeze

        if pretrained:
            # the new naming of pretrained weights, you can change to V2 if desired.
            weights = "IMAGENET1K_V1"
        else:
            weights = None

        if "swsl" in model_name or "ssl" in model_name:
            # These are the semi supervised and weakly semi supervised weights from Facebook
            self.model = torch.hub.load(
                "facebookresearch/semi-supervised-ImageNet1K-models", model_name
            )
        else:
            if "resnext50" in model_name:
                self.model = torchvision.models.resnext50_32x4d(weights=weights)
            elif "resnet50" in model_name:
                self.model = torchvision.models.resnet50(weights=weights)
            elif "101" in model_name:
                self.model = torchvision.models.resnet101(weights=weights)
            elif "152" in model_name:
                self.model = torchvision.models.resnet152(weights=weights)
            elif "34" in model_name:
                self.model = torchvision.models.resnet34(weights=weights)
            elif "18" in model_name:
                self.model = torchvision.models.resnet18(weights=weights)
            elif "wide_resnet50_2" in model_name:
                self.model = torchvision.models.wide_resnet50_2(weights=weights)
            else:
                raise NotImplementedError("Backbone architecture not recognized!")

        # freeze only if the model is pretrained
        if pretrained:
            if layers_to_freeze >= 0:
                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)
            if layers_to_freeze >= 1:
                self.model.layer1.requires_grad_(False)
            if layers_to_freeze >= 2:
                self.model.layer2.requires_grad_(False)
            if layers_to_freeze >= 3:
                self.model.layer3.requires_grad_(False)

        # remove the avgpool and most importantly the fc layer
        self.model.avgpool = None
        self.model.fc = None

        if 4 in layers_to_crop:
            self.model.layer4 = None
        if 3 in layers_to_crop:
            self.model.layer3 = None

        out_channels = 2048
        if "34" in model_name or "18" in model_name:
            out_channels = 512

        self.out_channels = (
            out_channels // 2 if self.model.layer4 is None else out_channels
        )
        self.out_channels = (
            self.out_channels // 2 if self.model.layer3 is None else self.out_channels
        )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        if self.model.layer3 is not None:
            x = self.model.layer3(x)
        if self.model.layer4 is not None:
            x = self.model.layer4(x)
        return x


class Resnet50gemModel(nn.Module):
    def __init__(self, fc_output_dim=2048):
        super().__init__()
        self.backbone = ResNet(
            model_name="resnet50",
            pretrained=True,
            layers_to_freeze=1,
            layers_to_crop=[],
        )

        self.aggregation = GeM()
        self.proj = nn.Linear(2048, fc_output_dim)
        self.norm = L2Norm()

    def forward(self, x: torch.Tensor, norm: bool = True) -> torch.Tensor:
        x = self.backbone(x)
        x = self.aggregation(x)
        if norm:
            x = self.norm(x)
        return x


class Resnet34gemModel(nn.Module):
    def __init__(self, fc_output_dim=2048):
        super().__init__()
        self.backbone = ResNet(
            model_name="resnet34",
            pretrained=True,
            layers_to_freeze=1,
            layers_to_crop=[],
        )

        self.aggregation = GeM()
        self.proj = nn.Linear(512, fc_output_dim)
        self.norm = L2Norm()

    def forward(self, x: torch.Tensor, norm: bool = True) -> torch.Tensor:
        x = self.backbone(x)
        x = self.aggregation(x)
        x = self.proj(x)
        if norm:
            x = self.norm(x)
        return x


preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((320, 320), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ResNet50_GeM(SingleStageBaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        self.model = Resnet50gemModel()
        name = "gem"
        weight_path = join(config["weights_directory"], name + ".ckpt")
        if pretrained:
            if not os.path.exists(weight_path):
                raise Exception(f"Could not find weights at {weight_path}")
            self.load_weights(weight_path)
            super().__init__(
                model=self.model,
                preprocess=preprocess,
                name=name,
                weight_path=weight_path,
            )
        else:
            super().__init__(
                model=self.model, preprocess=preprocess, name=name, weight_path=None
            )


class ResNet34_GeM(SingleStageBaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        self.model = Resnet34gemModel()
        name = "resnet34_gem"
        weight_path = join(config["weights_directory"], name + ".ckpt")
        if pretrained:
            if not os.path.exists(weight_path):
                raise Exception(f"Could not find weights at {weight_path}")
            self.load_weights(weight_path)
            super().__init__(
                model=self.model,
                preprocess=preprocess,
                name=name,
                weight_path=weight_path,
            )
        else:
            super().__init__(
                model=self.model, preprocess=preprocess, name=name, weight_path=None
            )
