import os
import sys

import torch
from torchvision import transforms

import PlaceRec.utils as utils
from PlaceRec.Methods import SingleStageBaseModelWrapper

preprocess = transforms.Compose(
    [
        transforms.Resize((480, 640)),  # (height, width)
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
            std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098],
        ),
    ]
)


class SFRS(SingleStageBaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        self.model = torch.hub.load(
            "yxgeee/OpenIBL", "vgg16_netvlad", pretrained=True
        ).eval()
        sys.stdout = original_stdout

        if not pretrained:
            self.model.apply(utils.init_weights)

        super().__init__(model=self.model, preprocess=preprocess, name="sfrs")
