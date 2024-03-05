import os
import pickle
import sys
from typing import Tuple

import numpy as np
import sklearn
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTModel
from os.path import join
from PlaceRec.utils import get_config
from .base_method import BaseModelWrapper

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
        return self.vit_model(x).last_hidden_state[:, 0, :]


preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((224, 224), antialias=True),
    ]
)


class ViT_CLS(BaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        name = "vit_cls"
        weight_path = join(config["weights_directory"], name + ".ckpt")

        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        model.encoder.layer = model.encoder.layer[:10]
        model = VitWrapper(model)
        if pretrained:
            if not os.path.exists(weight_path):
                raise Exception(f"Could not find weights at {weight_path}")

            if torch.cuda.is_available == "cuda":
                state_dict = torch.load(weight_path)["model_state_dict"]
            else:
                state_dict = torch.load(weight_path, map_location=torch.device("cpu"))[
                    "model_state_dict"
                ]
        state_dict = rename_state_dict(state_dict, "module.backbone.", "")
        true_state_dict = model.vit_model.state_dict()
        model.vit_model.load_state_dict(state_dict)
        super().__init__(model=model, preprocess=preprocess, name=name)
