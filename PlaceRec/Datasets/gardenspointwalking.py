import glob
import os
import zipfile

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.signal import convolve2d
from torch.utils.data import DataLoader

from ..utils import ImageDataset, s3_bucket_download
from .base_dataset import BaseDataset

package_directory = os.path.dirname(os.path.abspath(__file__))


class GardensPointWalking(BaseDataset):
    def __init__(self):
        # check to see if dataset is downloaded
        if not os.path.isdir(package_directory + "/raw_images/GardensPointWalking"):
            # download dataset as zip file
            s3_bucket_download("placerecdata/datasets/GardensPointWalking.zip", package_directory + "/raw_images/GardensPointWalking.zip")
            # unzip the dataset
            with zipfile.ZipFile(package_directory + "/raw_images/GardensPointWalking.zip", "r") as zip_ref:
                os.makedirs(package_directory + "/raw_images/GardensPointWalking")
                zip_ref.extractall(package_directory + "/raw_images/")

        # load images
        self.map_paths = np.array(sorted(glob.glob(package_directory + "/raw_images/GardensPointWalking/night_right/*")))
        self.query_paths = np.array(sorted(glob.glob(package_directory + "/raw_images/GardensPointWalking/day_right/*")))

        self.name = "gardenspointwalking"

    def query_images(
        self,
        partition: str,
        preprocess: torchvision.transforms.transforms.Compose = None,
    ) -> np.ndarray:
        size = len(self.query_paths)

        # get the required partition of the dataset
        if partition == "train":
            paths = self.query_paths[: int(size * 0.6)]
        elif partition == "val":
            paths = self.query_paths[int(size * 0.6) : int(size * 0.8)]
        elif partition == "test":
            paths = self.query_paths[int(size * 0.8) :]
        elif partition == "all":
            paths = self.query_paths
        else:
            raise Exception("Partition must be 'train', 'val' or 'all'")

        if preprocess == None:
            return np.array([np.array(Image.open(pth)) for pth in paths])
        else:
            imgs = np.array([np.array(Image.open(pth)) for pth in paths])
            return torch.stack([preprocess(q) for q in imgs])

    def map_images(
        self,
        partition: str,
        preprocess: torchvision.transforms.transforms.Compose = None,
    ) -> np.ndarray:
        if preprocess == None:
            return np.array([np.array(Image.open(pth)) for pth in self.map_paths])
        else:
            imgs = np.array([np.array(Image.open(pth)) for pth in self.map_paths])
            return torch.stack([preprocess(q) for q in imgs])

    def query_images_loader(
        self,
        partition: str,
        batch_size: int = 16,
        shuffle: bool = False,
        preprocess: torchvision.transforms.transforms.Compose = None,
        pin_memory: bool = False,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        size = len(self.query_paths)

        # get the required partition of the dataset
        if partition == "train":
            paths = self.query_paths[: int(size * 0.6)]
        elif partition == "val":
            paths = self.query_paths[int(size * 0.6) : int(size * 0.8)]
        elif partition == "test":
            paths = self.query_paths[int(size * 0.8) :]
        elif partition == "all":
            paths = self.query_paths
        else:
            raise Exception("Partition must be 'train', 'val' or 'all'")

        # build the dataloader
        dataset = ImageDataset(paths, preprocess=preprocess)
        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        return dataloader

    def map_images_loader(
        self,
        partition: str,
        batch_size: int = 16,
        shuffle: bool = False,
        preprocess: torchvision.transforms.transforms.Compose = None,
        pin_memory: bool = False,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        # build the dataloader
        dataset = ImageDataset(self.map_paths, preprocess=preprocess)
        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        return dataloader

    def ground_truth(self, partition: str, gt_type: str) -> np.ndarray:
        size = len(self.query_paths)

        gt = np.eye(len(self.map_paths)).astype("bool")

        # load the full grount truth matrix with the relevant form
        if gt_type == "soft":
            gt = convolve2d(gt.astype(int), np.ones((17, 1), "int"), mode="same").astype("bool")
        elif gt_type == "hard":
            pass
        else:
            raise Exception("gt_type must be either 'hard' or 'soft'")

        # select the relevant part of the ground truth matrix
        if partition == "train":
            gt = gt[:, : int(size * 0.6)]
        elif partition == "val":
            gt = gt[:, int(size * 0.6) : int(size * 0.8)]
        elif partition == "test":
            gt = gt[:, int(size * 0.8) :]
        elif partition == "all":
            pass
        else:
            raise Exception("partition must be either 'train', 'val', 'test' or 'all'")
        return gt.astype(bool)


if __name__ == "__main__":
    ds = GardensPointWalking()
