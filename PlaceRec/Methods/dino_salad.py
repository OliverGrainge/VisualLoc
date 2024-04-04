import os
import sys

import torch
from torchvision import transforms

import PlaceRec.utils as utils
from PlaceRec.Methods import SingleStageBaseModelWrapper

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class DinoSalad(SingleStageBaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        model = torch.hub.load("serizba/salad", "dinov2_salad")

        super().__init__(model=model, preprocess=preprocess, name="dinosalad")
