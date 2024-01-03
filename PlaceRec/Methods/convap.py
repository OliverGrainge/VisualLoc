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

from ..utils import get_config, s3_bucket_download
from .base_method import BaseModelWrapper

package_directory = os.path.dirname(os.path.abspath(__file__))
config = get_config()


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
            self.model = torch.hub.load("facebookresearch/semi-supervised-ImageNet1K-models", model_name)
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
                # self.model = torchvision.models.resnet18(pretrained=False)
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

        self.out_channels = out_channels // 2 if self.model.layer4 is None else out_channels
        self.out_channels = self.out_channels // 2 if self.model.layer3 is None else self.out_channels

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


class ConvAPModel(nn.Module):
    """Implementation of ConvAP as of https://arxiv.org/pdf/2210.10239.pdf

    Args:
        in_channels (int): number of channels in the input of ConvAP
        out_channels (int, optional): number of channels that ConvAP outputs. Defaults to 512.
        s1 (int, optional): spatial height of the adaptive average pooling. Defaults to 2.
        s2 (int, optional): spatial width of the adaptive average pooling. Defaults to 2.
    """

    def __init__(self, in_channels, out_channels=512, s1=2, s2=2):
        super(ConvAPModel, self).__init__()
        self.channel_pool = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)
        self.AAP = nn.AdaptiveAvgPool2d((s1, s2))

    def forward(self, x):
        x = self.channel_pool(x)
        x = self.AAP(x)
        x = F.normalize(x.flatten(1), p=2, dim=1)
        return x


def get_backbone(
    backbone_arch="resnet50",
    pretrained=True,
    layers_to_freeze=2,
    layers_to_crop=[],
):
    """Helper function that returns the backbone given its name

    Args:
        backbone_arch (str, optional): . Defaults to 'resnet50'.
        pretrained (bool, optional): . Defaults to True.
        layers_to_freeze (int, optional): . Defaults to 2.
        layers_to_crop (list, optional): This is mostly used with ResNet where we sometimes need to crop the last residual block (ex. [4]). Defaults to [].

    Returns:
        model: the backbone as a nn.Model object
    """
    if "resnet" in backbone_arch.lower():
        return ResNet(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)

    elif "efficient" in backbone_arch.lower():
        raise NotImplementedError

    elif "swin" in backbone_arch.lower():
        raise NotImplementedError


def get_aggregator(agg_arch="ConvAP", agg_config={}):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator. Defaults to 'ConvAP'.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.

    Returns:
        nn.Module: the aggregation layer
    """

    if "cosplace" in agg_arch.lower():
        raise NotImplementedError

    elif "gem" in agg_arch.lower():
        raise NotImplementedError

    elif "convap" in agg_arch.lower():
        assert "in_channels" in agg_config
        return ConvAPModel(**agg_config)


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.
    """

    def __init__(
        self,
        # ---- Backbone
        backbone_arch="resnet50",
        pretrained=True,
        layers_to_freeze=1,
        layers_to_crop=[],
        # ---- Aggregator
        agg_arch="ConvAP",  # CosPlace, NetVLAD, GeM, AVG
        agg_config={},
        # ---- Train hyperparameters
        lr=0.03,
        optimizer="sgd",
        weight_decay=1e-3,
        momentum=0.9,
        warmpup_steps=500,
        milestones=[5, 10, 15],
        lr_mult=0.3,
        # ----- Loss
        loss_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",
        miner_margin=0.1,
        faiss_gpu=False,
    ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin

        self.save_hyperparameters()  # write hyperparams into a file

        self.batch_acc = []  # we will keep track of the % of trivial pairs/triplets at the loss level

        self.faiss_gpu = faiss_gpu

        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        self.aggregator = get_aggregator(agg_arch, agg_config)

    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x


######################################### CONVAP MODEL ########################################################



# Note that images must be resized to 320x320
preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(
            (480, 640),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ConvAP(BaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        if pretrained:
            if not os.path.exists(join(config["weights_directory"], "resnet50_ConvAP_1024_2x2.ckpt")):
                raise Exception(f'Could not find weights at {join(config["weights_directory"], "resnet50_ConvAP_1024_2x2.ckpt")}')
            model = VPRModel(backbone_arch='resnet50', 
                            layers_to_crop=[],
                            agg_arch='ConvAP',
                            agg_config={'in_channels': 2048,
                                        'out_channels': 1024,
                                        's1' : 2,
                                        's2' : 2},
                            )
            if not torch.cuda.is_available():
                state_dict = torch.load(join(config["weights_directory"], "resnet50_ConvAP_1024_2x2.ckpt"), map_location=torch.device("cpu"))
            else: 
                state_dict = torch.load(join(config["weights_directory"], "resnet50_ConvAP_1024_2x2.ckpt"))

            model.load_state_dict(state_dict)
        else:
            model = VPRModel(backbone_arch='resnet50', 
                            layers_to_crop=[],
                            agg_arch='ConvAP',
                            agg_config={'in_channels': 2048,
                                        'out_channels': 1024,
                                        's1' : 2,
                                        's2' : 2},
                            )

        super().__init__(model=model, preprocess=preprocess, name="convap")

        # some layers not implemented on metal
        if self.device == "mps":
            self.device = "cpu"

        self.model.to(self.device)
