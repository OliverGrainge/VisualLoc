import os
import zipfile
from glob import glob

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.signal import convolve2d
from torch.utils.data import DataLoader
from os.path import join

from PlaceRec.Datasets.base_dataset import BaseDataset
from PlaceRec.utils import ImageIdxDataset, s3_bucket_download, get_config

config = get_config()
package_directory = os.path.dirname(os.path.abspath(__file__))


class ESSEX3IN1(BaseDataset):
    def __init__(self):
        if not join(config["datasets_directory"], "ESSEX3IN1_dataset"):
            raise Exception("Could Not Locate Dataset ESSEX3IN1_dataset")
    
        self.root = join(config["datasets_directory"], "ESSEX3IN1_dataset")

        self.map_paths = np.array([join(self.root, pth) for pth in np.load(join(self.root, "ESSEX_dbImages.npy"))])
        self.query_paths = np.array([join(self.root, pth) for pth in np.load(join(self.root, "ESSEX_qImages.npy"))])
        self.gt = np.load(join(self.root, "ESSEX_gt.npy"), allow_pickle=True)
        self.name = "essex3in1"


    def query_images_loader(
        self,
        batch_size: int = 16,
        shuffle: bool = False,
        preprocess: torchvision.transforms.transforms.Compose = None,
        pin_memory: bool = False,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        size = len(self.query_paths)


        # build the dataloader
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
        # build the dataloader
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
        return self.gt
