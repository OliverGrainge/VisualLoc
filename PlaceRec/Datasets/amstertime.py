from os.path import join
import os
from glob import glob
import numpy as np
from torchvision import transforms
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch.utils import data
import torchvision
from PlaceRec.utils import ImageIdxDataset
from torch.utils.data import DataLoader


class AmsterTime(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache)."""

    def __init__(
        self,
        datasets_folder="/Users/olivergrainge/Documents/github/datasets/VPR-datasets-downloader/datasets",
        dataset_name="amstertime",
        split="test",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.name = dataset_name
        self.dataset_folder = join(datasets_folder, dataset_name, "images", split)
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        self.test_method = "hard_resize"
        self.resize = (320, 320)

        #### Read paths and UTM coordinates for all images.
        database_folder = join(self.dataset_folder, "database")
        queries_folder = join(self.dataset_folder, "queries")
        if not os.path.exists(database_folder):
            raise FileNotFoundError(f"Folder {database_folder} does not exist")
        if not os.path.exists(queries_folder):
            raise FileNotFoundError(f"Folder {queries_folder} does not exist")
        self.database_paths = sorted(
            glob(join(database_folder, "**", "*.jpg"), recursive=True)
        )
        self.queries_paths = sorted(
            glob(join(queries_folder, "**", "*.jpg"), recursive=True)
        )
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]
        ).astype(np.float32)
        self.queries_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]
        ).astype(np.float32)

        # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.soft_positives_per_query = knn.radius_neighbors(
            self.queries_utms, radius=25, return_distance=False
        )

        self.images_paths = list(self.database_paths) + list(self.queries_paths)

        self.query_paths = self.queries_paths
        self.map_paths = self.database_paths

        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >"

    def map_images_loader(
        self,
        batch_size: int = 16,
        shuffle: bool = False,
        preprocess: torchvision.transforms.transforms.Compose = None,
        pin_memory: bool = False,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        dataset = ImageIdxDataset(self.database_paths, preprocess=preprocess)

        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        return dataloader

    def query_images_loader(
        self,
        batch_size: int = 16,
        shuffle: bool = False,
        preprocess: torchvision.transforms.transforms.Compose = None,
        pin_memory: bool = False,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        # build the dataloader
        dataset = ImageIdxDataset(self.queries_paths, preprocess=preprocess)
        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        return dataloader

    def ground_truth(self):
        return self.soft_positives_per_query
