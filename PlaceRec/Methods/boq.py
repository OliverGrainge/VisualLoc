import os
from os.path import join

import torch
import torchvision.transforms as T

import PlaceRec.utils as utils

from ..utils import get_config
from .base_method import SingleStageBaseModelWrapper

package_directory = os.path.dirname(os.path.abspath(__file__))
config = get_config()


import torch
import torchvision


class BoQBlock(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super(BoQBlock, self).__init__()

        self.encoder = torch.nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=nheads,
            dim_feedforward=4 * in_dim,
            batch_first=True,
            dropout=0.0,
        )
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))

        # the following two lines are used during training only, you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(
            in_dim, num_heads=nheads, batch_first=True
        )
        self.norm_q = torch.nn.LayerNorm(in_dim)
        #####

        self.cross_attn = torch.nn.MultiheadAttention(
            in_dim, num_heads=nheads, batch_first=True
        )
        self.norm_out = torch.nn.LayerNorm(in_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)

        q = self.queries.repeat(B, 1, 1)

        # the following two lines are used during training.
        # for stability purposes
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        #######

        out, attn = self.cross_attn(q, x, x)
        out = self.norm_out(out)
        return x, out, attn.detach()


class BoQ(torch.nn.Module):
    def __init__(
        self,
        in_channels=1024,
        proj_channels=512,
        num_queries=32,
        num_layers=2,
        row_dim=32,
    ):
        super().__init__()
        self.proj_c = torch.nn.Conv2d(
            in_channels, proj_channels, kernel_size=3, padding=1
        )
        self.norm_input = torch.nn.LayerNorm(proj_channels)

        in_dim = proj_channels
        self.boqs = torch.nn.ModuleList(
            [
                BoQBlock(in_dim, num_queries, nheads=in_dim // 64)
                for _ in range(num_layers)
            ]
        )

        self.fc = torch.nn.Linear(num_layers * num_queries, row_dim)

    def forward(self, x):
        # reduce input dimension using 3x3 conv when using ResNet
        x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)

        outs = []
        attns = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)

        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns


class ResNet(torch.nn.Module):
    def __init__(
        self,
        backbone_name="resnet50",
        crop_last_block=True,
    ):
        super().__init__()

        self.crop_last_block = crop_last_block

        if "18" in backbone_name:
            model = torchvision.models.resnet18()
        elif "34" in backbone_name:
            model = torchvision.models.resnet34()
        elif "50" in backbone_name:
            model = torchvision.models.resnet50()
        elif "101" in backbone_name:
            model = torchvision.models.resnet101()
        else:
            raise NotImplementedError("Backbone architecture not recognized!")

        # create backbone with only the necessary layers
        self.net = torch.nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            *([] if crop_last_block else [model.layer4]),
        )

        # calculate output channels
        out_channels = 2048
        if "34" in backbone_name or "18" in backbone_name:
            out_channels = 512

        self.out_channels = out_channels // 2 if crop_last_block else out_channels

    def forward(self, x):
        x = self.net(x)
        return x


class VPRModel(torch.nn.Module):
    def __init__(self, backbone, aggregator):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator

    def forward(self, x):
        x = self.backbone(x)
        x, attns = self.aggregator(x)
        return x, attns


AVAILABLE_BACKBONES = {
    # this list will be extended
    # "resnet18": [8192 , 4096],
    "resnet50": [16384],
    "dinov2": [12288],
}

MODEL_URLS = {
    "resnet50_16384": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/resnet50_16384.pth",
    "dinov2_12288": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/dinov2_12288.pth",
    # "resnet50_4096": "",
}


def get_trained_boq(backbone_name="resnet50", output_dim=16384):
    if backbone_name not in AVAILABLE_BACKBONES:
        raise ValueError(
            f"backbone_name should be one of {list(AVAILABLE_BACKBONES.keys())}"
        )
    try:
        output_dim = int(output_dim)
    except:
        raise ValueError(f"output_dim should be an integer, not a {type(output_dim)}")
    if output_dim not in AVAILABLE_BACKBONES[backbone_name]:
        raise ValueError(
            f"output_dim should be one of {AVAILABLE_BACKBONES[backbone_name]}"
        )

    if "resnet" in backbone_name:
        backbone = ResNet(
            backbone_name=backbone_name,
            crop_last_block=True,
        )
        aggregator = BoQ(
            in_channels=backbone.out_channels,  # make sure the backbone has out_channels attribute
            proj_channels=512,
            num_queries=64,
            num_layers=2,
            row_dim=output_dim // 512,  # 32 for resnet
        )
        # (self, in_channels=1024, proj_channels=512, num_queries=32, num_layers=2, row_dim=32)

    vpr_model = VPRModel(backbone=backbone, aggregator=aggregator)

    def new_forward(self, x):
        x, _ = self.aggregator(self.backbone(x))
        return x

    vpr_model.forward = new_forward.__get__(vpr_model, VPRModel)

    vpr_model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            MODEL_URLS[f"{backbone_name}_{output_dim}"],
            map_location=torch.device("cpu"),  # Ensure the model is loaded on CPU
        )
    )
    return vpr_model


preprocess = T.Compose(
    [
        T.ToTensor(),
        T.Resize((384, 384), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ResNet50_BoQ(SingleStageBaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        name = "boq"
        self.model = get_trained_boq(backbone_name="resnet50", output_dim=16384)

        if not pretrained:
            self.model.apply(utils.init_weights)
        else:
            super().__init__(
                model=self.model, preprocess=preprocess, name=name, weight_path=None
            )
