import os
import zipfile
from glob import glob
from os.path import join
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader

from PlaceRec.Datasets.base_dataset import BaseDataset
from PlaceRec.utils import ImageIdxDataset, s3_bucket_download, get_config

package_directory = os.path.dirname(os.path.abspath(__file__))
config = get_config()

class SFU(BaseDataset):
    def __init__(self):
        # check to see if dataset is downloaded
        if not os.path.isdir(join(config["datasets_directory"], "SFU")):
            # download dataset as zip file
            raise Exception("SFU is not Downloaded")

        # load images
        self.map_paths = np.array(sorted(glob(join(config["datasets_directory"] + "/raw_images/SFU/dry/*.jpg"))))
        self.query_paths = np.array(sorted(glob(join(config["datasets_directory"], "/raw_images/SFU/jan/*.jpg"))))

        self.name = "sfu"


    def query_images_loader(
        self,
        batch_size: int = 16,
        shuffle: bool = False,
        preprocess: torchvision.transforms.transforms.Compose = None,
        pin_memory: bool = False,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:

        dataset = ImageIdxDataset(self.query_paths, preprocess=preprocess)
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
        batch_size: int = 16,
        shuffle: bool = False,
        preprocess: torchvision.transforms.transforms.Compose = None,
        pin_memory: bool = False,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:

        dataset = ImageIdxDataset(self.map_paths, preprocess=preprocess)
        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        return dataloader

    def ground_truth(self) -> list:
        query_images = [img.split("/")[-1] for img in self.query_images]
        map_images = [img.split("/")[-1] for img in self.map_images]

        # Create a dictionary mapping image names to a list of their indices in map_images
        map_dict = {}
        for idx, img in enumerate(map_images):
            map_dict.setdefault(img, []).append(idx)

        # Get the indices using the dictionary
        ground_truth = [map_dict.get(query, []) for query in query_images]
        return ground_truth
