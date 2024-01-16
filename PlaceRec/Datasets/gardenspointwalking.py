import os
import zipfile
from glob import glob
from os.path import join

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.signal import convolve2d
from torch.utils.data import DataLoader

from PlaceRec.Datasets.base_dataset import BaseDataset
from PlaceRec.utils import ImageIdxDataset, get_config

package_directory = os.path.dirname(os.path.abspath(__file__))
config = get_config()


class GardensPointWalking(BaseDataset):
    def __init__(self):
        if not os.path.isdir(join(config["datasets_directory"], "GardensPointWalking")):
            raise Exception("GardensPointWalking dataset is not downloaded")

        # load images
        self.map_paths = np.array(
            sorted(
                glob(
                    config["datasets_directory"] + "/GardensPointWalking/night_right/*"
                )
            )
        )
        self.query_paths = np.array(
            sorted(
                glob(config["datasets_directory"] + "/GardensPointWalking/day_right/*")
            )
        )

        self.name = "gardenspointwalking"

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
        query_images = [img.split("/")[-1] for img in self.query_paths]
        map_images = [img.split("/")[-1] for img in self.map_paths]

        print(len(query_images), len(map_images))

        # Create a dictionary mapping image names to a list of their indices in map_images
        map_dict = {}
        for idx, img in enumerate(map_images):
            map_dict.setdefault(img, []).append(idx)

        # Get the indices using the dictionary
        ground_truth = [np.array(map_dict.get(query, [])) for query in query_images]
        print(len(ground_truth))
        return ground_truth


if __name__ == "__main__":
    ds = GardensPointWalking()
