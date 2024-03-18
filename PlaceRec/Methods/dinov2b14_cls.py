import os
import sys

import torch
from torchvision import transforms

import PlaceRec.utils as utils
from PlaceRec.Methods import BaseModelWrapper

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize
    ]
)


class DINOv2B14_CLS(BaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        sys.stdout = original_stdout

        super().__init__(model=model, preprocess=preprocess, name="dinov2b14_cls")
