import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pathlib
import boto3
import os
from tqdm import tqdm

import boto3
import botocore







# Download the file





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


# ============== S3 Bucket ===========================================================================

class ProgressPercentage(tqdm):
    def __init__(self, client, bucket, filename):
        self._size = client.head_object(Bucket=bucket, Key=filename)["ContentLength"]
        super(ProgressPercentage, self).__init__(total=self._size, unit='B', unit_scale=True, desc="Downloading " + filename.split('/')[-1])

    def __call__(self, bytes_amount):
        self.update(bytes_amount)

def s3_bucket_download(remote_path: str, local_path: str):
    s3 = boto3.client('s3', region_name="eu-north-1", config=boto3.session.Config(signature_version=botocore.UNSIGNED))
    
    # Define the bucket name and the datasets to download
    bucket_name = 'visuallocbucket'
    
    # Download each dataset
    progress = ProgressPercentage(s3, bucket_name, remote_path)
    s3.download_file(bucket_name, remote_path, local_path, Callback=progress)
    
    return None

# ==========================================================================================================


def get_dataset(name: str = None):
    if name == "gardenspointwalking":
        from PlaceRec.Datasets import GardensPointWalking

        dataset = GardensPointWalking()
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


def get_method(name: str = None):
    if name == "calc":
        from PlaceRec.Methods import CALC

        method = CALC()
    elif name == "netvlad":
        from PlaceRec.Methods import NetVLAD

        method = NetVLAD()
    elif name == "hog":
        from PlaceRec.Methods import HOG

        method = HOG()
    elif name == "cosplace":
        from PlaceRec.Methods import CosPlace

        method = CosPlace()
    elif name == "alexnet":
        from PlaceRec.Methods import AlexNet

        method = AlexNet()
    elif name == "hybridnet":
        from PlaceRec.Methods import HybridNet

        method = HybridNet()
    elif name == "amosnet":
        from PlaceRec.Methods import AmosNet

        method = AmosNet()
    elif name == "convap":
        from PlaceRec.Methods import CONVAP

        method = CONVAP()
    elif name == "mixvpr":
        from PlaceRec.Methods import MixVPR

        method = MixVPR()
    elif name == "regionvlad":
        from PlaceRec.Methods import RegionVLAD

        method = RegionVLAD()
    else:
        raise Exception("Method not implemented")
    return method
