import zipfile
import os
import numpy as np
from .base_dataset import BaseDataset
import torchvision
import torch
from glob import glob
from PIL import Image
from ..utils import ImageDataset
from torch.utils.data import DataLoader
from scipy.signal import convolve2d


package_directory = os.path.dirname(os.path.abspath(__file__))


class ESSEX3IN1(BaseDataset):
    def __init__(self):
        if not os.path.isdir(package_directory + "/raw_images/ESSEX3IN1"):
            raise Exception("Dataset not downloaded")

        self.query_paths = np.array(
            sorted(glob(package_directory + "/raw_images/ESSEX3IN1/query_combined/*"))
        )
        self.map_paths = np.array(
            sorted(
                glob(package_directory + "/raw_images/ESSEX3IN1/reference_combined/*")
            )
        )

        self.name = "essex3in1"

    def query_images(
        self,
        partition: str,
        preprocess: torchvision.transforms.transforms.Compose = None,
    ) -> torch.Tensor:
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
            return np.array(
                [np.array(Image.open(pth).resize((720, 720))) for pth in paths]
            )
        else:
            imgs = np.array(
                [np.array(Image.open(pth).resize((720, 720))) for pth in paths]
            )
            return torch.stack([preprocess(q) for q in imgs])

    def map_images(
        self,
        partition: str,
        preprocess: torchvision.transforms.transforms.Compose = None,
    ) -> torch.Tensor:
        if preprocess == None:
            return np.array(
                [np.array(Image.open(pth).resize((720, 720))) for pth in self.map_paths]
            )
        else:
            imgs = np.array(
                [np.array(Image.open(pth).resize((720, 720))) for pth in self.map_paths]
            )
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
            # No soft ground truth for this dataset
            pass
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
