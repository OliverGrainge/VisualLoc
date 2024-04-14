import os
import pathlib
from argparse import Namespace
from typing import Union

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import Dataset
from tqdm import tqdm


class ImageIdxDataset(Dataset):
    def __init__(self, img_paths, preprocess=None):
        self.img_paths = img_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.preprocess is not None:
            img = np.array(Image.open(self.img_paths[idx]))[:, :, :3]
            img = Image.fromarray(img)
            img = self.preprocess(img)
            return idx, img

        img = np.array(Image.open(self.img_paths[idx]).resize((224, 224)))[:, :, :3]
        return idx, img


def get_dataset(name: str = None):
    module_name = "PlaceRec.Datasets"
    method_module = __import__(module_name, fromlist=[name])
    method_class = getattr(method_module, name)
    return method_class()


def get_method(name: str = None, pretrained: bool = True):
    module_name = "PlaceRec.Methods"
    method_module = __import__(module_name, fromlist=[name])
    method_class = getattr(method_module, name)
    return method_class(pretrained=pretrained)


def cosine_distance(x1, x2):
    cosine_sim = nn.CosineSimilarity(dim=0)(x1, x2)
    return 1 - cosine_sim


def get_loss_function(args):
    if args.loss_distance == "l2":
        return nn.TripletMarginLoss(args.margin, p=2, reduction="sum")
    elif args.loss_distance == "cosine":
        return nn.TripletMarginWithDistanceLoss(
            distance_function=cosine_distance, margin=args.margin
        )


def get_config():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Example: Kaiming Initialization for Conv2D and Linear layers
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def get_logger():
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        filename="VisualLoc.log",
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
    )
    return logging
