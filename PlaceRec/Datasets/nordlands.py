import os
import zipfile
from glob import glob

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.signal import convolve2d
from torch.utils.data import DataLoader
from tqdm import tqdm

from PlaceRec.Datasets.base_dataset import BaseDataset
from PlaceRec.utils import ImageIdxDataset, s3_bucket_download

package_directory = os.path.dirname(os.path.abspath(__file__))

QUERY_SET = ["summer"]
MAP_SET = ["fall"]


def image_idx(img_path: str):
    img_path = int(img_path.split("/")[-1][:-4])
    return img_path


def get_paths(partition: list, seasons: list) -> list:
    if not os.path.isdir(package_directory + "/raw_images/Nordlands"):
        raise Exception("Please Download Nordlands dataset to /raw_images/Nordlands")

    root = package_directory + "/raw_images/Nordlands/" + partition
    images = []
    for season in seasons:
        pth = root + "/" + season + "_images_" + partition
        sections = glob(pth + "/*")
        for section in sections:
            images += glob(section + "/*")
    images = np.array(images)
    image_index = [image_idx(img) for img in images]
    sort_idx = np.argsort(image_index)
    images = images[sort_idx]
    return images


class Nordlands(BaseDataset):
    def __init__(self):
        self.train_map_paths = get_paths("train", MAP_SET)
        self.train_query_paths = get_paths("train", QUERY_SET)
        self.test_map_paths = get_paths("test", MAP_SET)
        self.test_query_paths = get_paths("test", QUERY_SET)

        self.query_paths = self.query_partition("all")
        self.map_paths = self.map_partition("all")

        self.name = "norldlands"

    def query_partition(self, partition: str) -> np.ndarray:
        # get the required partition of the dataset
        if partition == "train":
            paths = self.train_query_paths[: int(len(self.train_query_paths) * 0.8)]
        elif partition == "val":
            paths = self.train_query_paths[int(len(self.train_query_paths) * 0.8) :]
        elif partition == "test":
            paths = self.test_query_paths
        elif partition == "all":
            paths = self.train_query_paths
        else:
            raise Exception("Partition must be 'train', 'val' or 'all'")
        return paths

    def map_partition(self, partition: str) -> np.ndarray:
        if partition == "train" or partition == "val":
            paths = self.train_map_paths
        elif partition == "test":
            paths = self.test_map_paths
        elif partition == "all":
            paths = self.train_map_paths
        else:
            raise Exception("Partition not found")
        return paths

    def query_images(
        self,
        partition: str,
        preprocess: torchvision.transforms.transforms.Compose = None,
    ) -> torch.Tensor:
        paths = self.query_partition(partition)

        if preprocess == None:
            return np.array([np.array(Image.open(pth).resize((320, 320)))[:, :, :3] for pth in paths])
        else:
            imgs = np.array([np.array(Image.open(pth).resize((320, 320)))[:, :, :3] for pth in paths])
            return torch.stack([preprocess(q) for q in imgs])

    def map_images(
        self,
        partition: str,
        preprocess: torchvision.transforms.transforms.Compose = None,
    ) -> torch.Tensor:
        paths = self.map_partition(partition)

        if preprocess == None:
            return np.array([np.array(Image.open(pth).resize((320, 320)))[:, :, :3] for pth in paths])
        else:
            imgs = np.array([np.array(Image.open(pth).resize((320, 320)))[:, :, :3] for pth in paths])
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
        paths = self.query_partition(partition)

        # build the dataloader
        dataset = ImageIdxDataset(paths, preprocess=preprocess)
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
        paths = self.map_partition(partition)
        dataset = ImageIdxDataset(paths, preprocess=preprocess)
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

        query_images = [img.split("/")[-1] for img in query_images]
        map_images = [img.split("/")[-1] for img in map_images]

        # Create a dictionary mapping image names to a list of their indices in map_images
        map_dict = {}
        for idx, img in enumerate(map_images):
            map_dict.setdefault(img, []).append(idx)

        # Get the indices using the dictionary
        ground_truth = [map_dict.get(query, []) for query in query_images]
        return ground_truth
