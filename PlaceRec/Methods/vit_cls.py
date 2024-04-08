import os
import pickle
import sys
from os.path import join
from typing import Tuple

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTModel

from PlaceRec.utils import get_config

from .base_method import SingleStageBaseModelWrapper

config = get_config()


def rename_state_dict(orig_dict, pattern1, pattern2) -> dict:
    new_dict = {}
    for key in list(orig_dict.keys()):
        new_key = key.replace(pattern1, pattern2)
        new_dict[new_key] = orig_dict[key]
    return new_dict


class VitWrapper(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit_model = vit_model

    def forward(self, x):
        x = self.vit_model(x).last_hidden_state[:, 0, :]
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x


preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((224, 224), antialias=True),
    ]
)


class ViT_CLS(SingleStageBaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        name = "vit_cls"
        weight_path = join(config["weights_directory"], name + ".ckpt")

        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model.encoder.layer = self.model.encoder.layer[:10]
        self.model = VitWrapper(self.model)

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
