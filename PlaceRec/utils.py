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


class ImageDataset(Dataset):
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
            return img

        img = np.array(Image.open(self.img_paths[idx]).resize((320, 320)))[:, :, :3]
        return img


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


# ==========================================================================================================


def get_dataset(name: str = None):
    name = name.lower()
    if name == "gardenspointwalking":
        from PlaceRec.Datasets import GardensPointWalking

        dataset = GardensPointWalking()
    elif name == "pitts250k":
        from PlaceRec.Datasets import Pitts250k

        dataset = Pitts250k()
    elif name == "mapillarysls":
        from PlaceRec.Datasets import MapillarySLS

        dataset = MapillarySLS()
    elif name == "pitts30k":
        from PlaceRec.Datasets import Pitts30k

        dataset = Pitts30k()
    elif name == "sfu":
        from PlaceRec.Datasets import SFU

        dataset = SFU()
    elif name == "essex3in1":
        from PlaceRec.Datasets import ESSEX3IN1

        dataset = ESSEX3IN1()
    elif name == "nordlands":
        from PlaceRec.Datasets import Nordlands

        dataset = Nordlands()
    elif name == "crossseasons":
        from PlaceRec.Datasets import CrossSeason

        dataset = CrossSeason()
    elif name == "spedtest":
        from PlaceRec.Datasets import SpedTest

        dataset = SpedTest()
    elif name == "inriaholidays":
        from PlaceRec.Datasets import InriaHolidays

        dataset = InriaHolidays()
    else:
        raise Exception("Dataset '" + name + "' not implemented")
    return dataset


def get_method(name: str = None, pretrained: bool = True):
    if name == "calc":
        from PlaceRec.Methods import CALC

        method = CALC(pretrained=pretrained)
    elif name == "eigenplaces":
        from PlaceRec.Methods import EigenPlaces

        method = EigenPlaces(pretrained=pretrained)
    elif name == "resnet34convap":
        from PlaceRec.Methods import ResNet34ConvAP

        method = ResNet34ConvAP(pretrained=pretrained)
    elif name == "netvlad":
        from PlaceRec.Methods import NetVLAD

        method = NetVLAD(pretrained=pretrained)
    elif name == "cosplace":
        from PlaceRec.Methods import CosPlace

        method = CosPlace(pretrained=pretrained)
    elif name == "hybridnet":
        from PlaceRec.Methods import HybridNet

        method = HybridNet(pretrained=pretrained)
    elif name == "amosnet":
        from PlaceRec.Methods import AmosNet

        method = AmosNet(pretrained=pretrained)
    elif name == "resnet50convap":
        from PlaceRec.Methods import ResNet50ConvAP

        method = ResNet50ConvAP(pretrained=pretrained)
    elif name == "mixvpr":
        from PlaceRec.Methods import MixVPR

        method = MixVPR(pretrained=pretrained)
    elif name == "resnet18_gem":
        from PlaceRec.Methods import ResNet18GeM

        method = ResNet18GeM(pretrained=pretrained)
    elif name == "resnet50_gem":
        from PlaceRec.Methods import ResNet50GeM

        method = ResNet50GeM(pretrained=pretrained)
    elif name == "cct384_netvlad":
        from PlaceRec.Methods import CCT384_NetVLAD

        method = CCT384_NetVLAD(pretrained=pretrained)
    else:
        raise Exception("Method not implemented")
    return method


def get_training_logger(args: Namespace, project_name: Union[str, None] = None):
    if args.logger_type.lower() == "wandb":
        logger = WandbLogger(project=project_name)
    elif args.logger_type.lower() == "tensorboard":
        logger = TensorBoardLogger("tb_logs", name=project_name)
    else:
        raise NotImplementedError()
    return logger


def cosine_distance(x1, x2):
    # Cosine similarity ranges from -1 to 1, so we add 1 to make it non-negative
    # and then normalize it to range from 0 to 1
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
