import zipfile
import os
import numpy as np
from .base_dataset import BaseDataset
import torchvision
import torch
from glob import glob
from PIL import Image
from ..utils import ImageDataset, s3_bucket_download
from torch.utils.data import DataLoader
import pandas as pd


package_directory = os.path.dirname(os.path.abspath(__file__))


class UAV_Round2(BaseDataset):
    def __init__(self):
        # check to see if dataset is downloaded
        if not os.path.isdir(package_directory + "/raw_images/SFU"):
            raise NotImplementedError("Dataset not been uploaded to s3 bucket")

        self.name = "sfu"

    def query_partition(self, partition: str) -> np.ndarray:
        if partition == "train":
            paths = sorted(
                glob(package_directory + "/raw_images/UAV/Train/query_images/*")
            )
        elif partition == "val":
            paths = sorted(
                glob(package_directory + "/raw_images/UAV/Val/query_images/*")
            )
        elif partition == "test":
            paths = sorted(
                glob(package_directory + "/raw_images/UAV/Val/query_images/*")
            )
        return np.array(paths)

    def map_partition(self, partition: str) -> np.ndarray:
        if partition == "train":
            paths = sorted(
                glob(
                    package_directory
                    + "/raw_images/UAV/Train/reference_images/offset_0_None/*"
                )
            )
        elif partition == "val":
            paths = sorted(
                glob(
                    package_directory
                    + "/raw_images/UAV/Val/reference_images/offset_0_None/*"
                )
            )
        elif partition == "test":
            paths = sorted(
                glob(
                    package_directory
                    + "/raw_images/UAV/Val/reference_images/offset_0_None/*"
                )
            )
        return np.array(paths)

    def query_images(
        self,
        partition: str,
        preprocess: torchvision.transforms.transforms.Compose = None,
    ) -> torch.Tensor:
        paths = self.query_partition(partition)

        if preprocess == None:
            return np.array(
                [
                    np.array(Image.open(pth).resize((320, 320)))[:, :, :3]
                    for pth in paths
                ]
            )
        else:
            imgs = np.array(
                [
                    np.array(Image.open(pth).resize((320, 320)))[:, :, :3]
                    for pth in paths
                ]
            )
            return torch.stack([preprocess(q) for q in imgs])

    def map_images(
        self,
        partition: str,
        preprocess: torchvision.transforms.transforms.Compose = None,
    ) -> torch.Tensor:
        paths = self.map_partition(partition)

        if preprocess == None:
            return np.array(
                [
                    np.array(Image.open(pth).resize((320, 320)))[:, :, :3]
                    for pth in paths
                ]
            )
        else:
            imgs = np.array(
                [
                    np.array(Image.open(pth).resize((320, 320)))[:, :, :3]
                    for pth in paths
                ]
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
        paths = self.query_partition(partition)

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
        paths = self.map_partition(partition)
        dataset = ImageDataset(paths, preprocess=preprocess)
        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        return dataloader

    def ground_truth(self, partition: str, gt_type="hard") -> np.ndarray:
        if partition == "train":
            df = pd.read_csv(package_directory + "/raw_images/UAV/Train/gt_matches.csv")

        elif partition == "test":
            df = pd.read_csv(package_directory + "/raw_images/UAV/Val/gt_matches.csv")

        elif partition == "val":
            df = pd.read_csv(package_directory + "/raw_images/UAV/Val/gt_matches.csv")

        elif partition == "all":
            raise NotImplementedError

        q_paths = self.query_partition(partition)
        m_paths = self.map_partition(partition)
        q_paths = [q_path.split("/")[-1] for q_path in q_paths]
        m_paths = [m_path.split("/")[-1] for m_path in m_paths]

        gt = np.zeros((len(m_paths), len(q_paths)), dtype=bool)

        gt_queries = df["query_name"].to_numpy()
        gt_maps = df["ref_name"].to_numpy()

        for ref_name in gt_maps:
            for query_name in gt_queries:
                query_idx = q_paths.index(query_name)
                map_idx = m_paths.index(ref_name)
                gt[map_idx, query_idx] = True

        return gt
