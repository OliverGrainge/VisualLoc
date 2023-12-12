import os
import zipfile
from glob import glob
from os.path import join

import numpy as np
import torch
import torchvision
import yaml
from PIL import Image
from scipy.signal import convolve2d
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

from PlaceRec.Datasets.base_dataset import BaseDataset
from PlaceRec.utils import ImageIdxDataset, s3_bucket_download

with open(join(os.getcwd(), "config.yaml"), "r") as file:
    config = yaml.safe_load(file)

package_directory = os.path.dirname(os.path.abspath(__file__))


class MSLS(BaseDataset):
    def __init__(self):
        # check to see if dataset is downloaded
        if not os.path.isdir(join(config["train"]["datasets_folder"], "msls", "images")):
            # download dataset as zip file
            raise Exception("MSLS Not Downloaded")

        self.dataset_folder = join(config["train"]["datasets_folder"], "msls", "images", "test")
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")

        #### Read paths and UTM coordinates for all images.
        database_folder = join(self.dataset_folder, "database")
        queries_folder = join(self.dataset_folder, "queries")

        if not os.path.exists(database_folder):
            raise FileNotFoundError(f"Folder {database_folder} does not exist")
        if not os.path.exists(queries_folder):
            raise FileNotFoundError(f"Folder {queries_folder} does not exist")

        self.test_database_paths = sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
        self.test_queries_paths = sorted(glob(join(queries_folder, "**", "*.jpg"), recursive=True))
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.test_database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.test_database_paths]).astype(float)
        self.test_queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.test_queries_paths]).astype(float)

        # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.test_database_utms)
        self.test_soft_positives_per_query = knn.radius_neighbors(
            self.test_queries_utms,
            radius=config["train"]["val_positive_dist_threshold"],
            return_distance=False,
        )
        self.test_images_paths = list(self.test_database_paths) + list(self.test_queries_paths)
        self.test_database_num = len(self.test_database_paths)
        self.test_queries_num = len(self.test_queries_paths)

        self.name = "msls"

    def query_partition(self, partition: str) -> np.ndarray:
        # get the required partition of the dataset
        if partition in ["val", "test"]:
            return self.test_queries_paths
        else:
            raise Exception("partition must be val or test, train is too large")

    def query_images(
        self,
        partition: str,
        preprocess: torchvision.transforms.transforms.Compose = None,
    ) -> np.ndarray:
        # get the required partition of the dataset
        paths = self.query_partition(partition)

        if preprocess == None:
            return np.array([np.array(Image.open(pth)) for pth in paths])
        else:
            imgs = np.array([np.array(Image.open(pth)) for pth in paths])
            return torch.stack([preprocess(q) for q in imgs])

    def map_images(self, partition: str, preprocess: torchvision.transforms.transforms.Compose = None):
        if partition == "train":
            raise Exception(" only test or val partitions available")

        elif partition == "val" or "test":
            if preprocess == None:
                return np.array([np.array(Image.open(pth)) for pth in self.test_database_paths])
            else:
                imgs = np.array([np.array(Image.open(pth)) for pth in self.test_database_paths])
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
        # get the required partition of the dataset
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
        # build the dataloader
        if partition == "train" or partition == "all":
            raise Exception("Only test of val partitions available")
        elif partition in ["test", "val"]:
            dataset = ImageIdxDataset(self.test_database_paths, preprocess=preprocess)
        else:
            raise Exception("partition must be train, test, val or all")

        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        return dataloader

    def ground_truth(self, partition: str) -> np.ndarray:
        if partition == "train":
            raise Exception("Onlly val or test partitions available")
        elif partition == "val" or partition == "test":
            return [list(matches) for matches in self.test_soft_positives_per_query]
        elif partition == "all":
            raise Exception("Onlly val or test partitions available")
        else:
            raise Exception("partition must be train, val, test or all")


if __name__ == "__main__":
    ds = MSLS()
