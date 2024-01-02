import os
import pathlib

import boto3
import botocore
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Union
from argparse import Namespace
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

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


# ============== S3 Bucket ===========================================================================


class ProgressPercentage(tqdm):
    def __init__(self, client, bucket, filename):
        self._size = client.head_object(Bucket=bucket, Key=filename)["ContentLength"]
        super(ProgressPercentage, self).__init__(total=self._size, unit="B", unit_scale=True, desc="Downloading " + filename.split("/")[-1])

    def __call__(self, bytes_amount):
        self.update(bytes_amount)


def s3_bucket_download(remote_path: str, local_path: str):
    s3 = boto3.client("s3", region_name="eu-north-1", config=boto3.session.Config(signature_version=botocore.UNSIGNED))

    # Define the bucket name and the datasets to download
    bucket_name = "visuallocbucket"

    # Download each dataset
    progress = ProgressPercentage(s3, bucket_name, remote_path)
    s3.download_file(bucket_name, remote_path, local_path, Callback=progress)

    return None


# ==========================================================================================================


def get_dataset(name: str = None):
    name = name.lower()
    if name == "gardenspointwalking":
        from PlaceRec.Datasets import GardensPointWalking

        dataset = GardensPointWalking()
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
    elif name == "netvlad":
        from PlaceRec.Methods import NetVLAD

        method = NetVLAD(pretrained=pretrained)
    elif name == "hog":
        from PlaceRec.Methods import HOG

        method = HOG(pretrained=pretrained)
    elif name == "cosplace":
        from PlaceRec.Methods import CosPlace

        method = CosPlace(pretrained=pretrained)
    elif name == "alexnet":
        from PlaceRec.Methods import AlexNet

        method = AlexNet(pretrained=pretrained)
    elif name == "hybridnet":
        from PlaceRec.Methods import HybridNet

        method = HybridNet(pretrained=pretrained)
    elif name == "amosnet":
        from PlaceRec.Methods import AmosNet

        method = AmosNet(pretrained=pretrained)
    elif name == "convap":
        from PlaceRec.Methods import ConvAP

        method = ConvAP(pretrained=pretrained)
    elif name == "mixvpr":
        from PlaceRec.Methods import MixVPR

        method = MixVPR(pretrained=pretrained)
    elif name == "regionvlad":
        from PlaceRec.Methods import RegionVLAD

        method = RegionVLAD(pretrained=pretrained)
    elif name == "resnet18_gem":
        from PlaceRec.Methods import ResNet18GeM

        method = ResNet18GeM(pretrained=pretrained)
    elif name == "resnet50_gem":
        from PlaceRec.Methods import ResNet50GeM

        method = ResNet50GeM(pretrained=pretrained)

    elif name == "vit_base_patch16_224_gap":
        from PlaceRec.Methods import ViT_base_patch16_224_gap

        method = ViT_base_patch16_224_gap(pretrained=pretrained)
    elif name == "dinov2_base_patch16_224_gap":
        from PlaceRec.Methods import DinoV2_base_patch16_224_gap

        method = DinoV2_base_patch16_224_gap(pretrained=pretrained)
    elif name == "cct384_netvlad":
        from PlaceRec.Methods import CCT384_NetVLAD

        method = CCT384_NetVLAD(pretrained=pretrained)
    elif name == "vit_base_patch16_224_cls":
        from PlaceRec.Methods import ViT_base_patch16_224_cls

        method = ViT_base_patch16_224_cls(pretrained=pretrained)
    else:
        raise Exception("Method not implemented")
    return method


def get_training_logger(config: dict, project_name: Union[str, None] = None):
    if config["train"]["logger"].lower() == "wandb": 
        logger = WandbLogger(project=project_name)
    elif config["train"]["logger"].lower() == "tensorboard":
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
        return nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=args.margin)


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
