import os
import zipfile
from glob import glob

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader

from ..utils import ImageDataset, s3_bucket_download
from .base_dataset import BaseDataset

package_directory = os.path.dirname(os.path.abspath(__file__))


class SFU(BaseDataset):
    def __init__(self):
        # check to see if dataset is downloaded
        if not os.path.isdir(package_directory + "/raw_images/SFU"):
            # download dataset as zip file
            s3_bucket_download("placerecdata/datasets/SFU.zip", package_directory + "/raw_images/SFU.zip")
            # unzip the dataset
            with zipfile.ZipFile(package_directory + "/raw_images/SFU.zip", "r") as zip_ref:
                os.makedirs(package_directory + "/raw_images/SFU")
                zip_ref.extractall(package_directory + "/raw_images/")

        # load images
        self.map_paths = np.array(sorted(glob(package_directory + "/raw_images/SFU/dry/*.jpg")))
        self.query_paths = np.array(sorted(glob(package_directory + "/raw_images/SFU/jan/*.jpg")))

        self.name = "sfu"
    
    def query_partition(self, partition: str) -> np.ndarray:
        # get the required partition of the dataset
        size = len(self.query_paths)
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
        return paths

    def map_partition(self, partition: str) -> np.ndarray:
        return self.map_paths

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

    def map_images(self, preprocess: torchvision.transforms.transforms.Compose = None):
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


    def ground_truth(self, partition: str) -> np.ndarray:
        query_images = self.query_partition(partition=partition)
        map_images = self.map_partition(partition)

        query_images = [img.split('/')[-1] for img in query_images]
        map_images = [img.split('/')[-1] for img in map_images]

        # Create a dictionary mapping image names to a list of their indices in map_images
        map_dict = {}
        for idx, img in enumerate(map_images):
            map_dict.setdefault(img, []).append(idx)

        # Get the indices using the dictionary
        ground_truth = [map_dict.get(query, []) for query in query_images]
        return ground_truth

