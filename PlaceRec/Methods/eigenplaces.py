import os
import sys

import torch
from torchvision import transforms

import PlaceRec.utils as utils
from PlaceRec.Methods import SingleStageBaseModelWrapper

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((480, 640), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class EigenPlaces(SingleStageBaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        self.model = torch.hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=2048,
        )
        sys.stdout = original_stdout

        # if not pretrained:
        #    model.apply(utils.init_weights)

        super().__init__(
            model=self.model,
            preprocess=preprocess,
            name="eigenplaces",
            weight_path=None,
        )
