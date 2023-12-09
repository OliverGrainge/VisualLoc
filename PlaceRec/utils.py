import os
import pathlib

import boto3
import botocore
import numpy as np
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F

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
            return img, idx

        img = np.array(Image.open(self.img_paths[idx]).resize((320, 320)))[:, :, :3]
        return img, idx


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
    if name == "gardenspointwalking":
        from PlaceRec.Datasets import GardensPointWalking

        dataset = GardensPointWalking()
    elif name == "pitts30k":
        from PlaceRec.Datasets import Pitts30k

        dataset = Pitts30k()
    elif name == "msls":
        from PlaceRec.Datasets import MSLS

        dataset = MSLS()
    elif name == "stlucia_large":
        from PlaceRec.Datasets import StLucia_large

        dataset = StLucia_large()
    elif name == "stlucia_small":
        from PlaceRec.Datasets import StLucia_small

        dataset = StLucia_small()
    elif name == "sfu":
        from PlaceRec.Datasets import SFU

        dataset = SFU()
    elif name == "essex3in1":
        from PlaceRec.Datasets import ESSEX3IN1

        dataset = ESSEX3IN1()
    elif name == "nordlands":
        from PlaceRec.Datasets import Nordlands

        dataset = Nordlands()
    elif name == "gsvcities":
        from PlaceRec.Datasets import GsvCities

        dataset = GsvCities()
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
    else:
        raise Exception("Method not implemented")
    return method


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