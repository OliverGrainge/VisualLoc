from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
import torchvision


class BaseDataset(ABC):
    """
    This is an abstract class that serves as a template for implementing
    visual place recognition datasets.

    Attributes:
        query_paths (np.ndarray): A vector of type string providing relative paths to the query images
        map_paths (np.ndarray): A vector of type string providing relative paths to the map images
        name (str): a name for the dataset
    """
    query_paths = None
    map_paths = None
    name = None


    @abstractmethod
    def query_images_loader(
        self,
        batch_size: int = 16,
        shuffle: bool = False,
        preprocess: torchvision.transforms.transforms.Compose = None,
        pin_memory: bool = False,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        """
        This function returns a torch dataloader that can be used for loading
        the query images in batches. The dataloader will reutn batches of [batch_size, H, W, C]
        where the datatype is uint8

        Args:
            partition (str): determines which partition the datasets query images to return.
                             must bet either "train", "val", "test", or "all"
            batch_size (int): The batch size for the dataloader to return
            shuffle: (bool): True if you want the order of query images to be shuffles.
                             False will keep the images in sequence.
            preprocess (torchvision.transforms.transforms.Compose): The image augmentations
                             to apply to the query images
            pin_memory (bool): pinning memory from cpu to gpu if using gpu for inference
            num_workers (int): number of worker used for proceesing the images by the dataloader.

        Returns:
            torch.utils.data.DataLoader: a dataloader for the query image set.

        """
        pass

    @abstractmethod
    def map_images_loader(
        self,
        batch_size: int = 16,
        shuffle: bool = False,
        preprocess: torchvision.transforms.transforms.Compose = None,
        pin_memory: bool = False,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        """
        This function returns a torch dataloader that can be used for loading
        the map images in batches. The dataloader will reutn batches of [batch_size, H, W, C]
        where the datatype is uint8

        Args:
            partition (str): determines which partition the datasets map images to return.
                             must bet either "train", "val", "test", or "all"
            batch_size (int): The batch size for the dataloader to return
            shuffle: (bool): True if you want the order of map images to be shuffles.
                             False will keep the images in sequence.
            preprocess (torchvision.transforms.transforms.Compose): The image augmentations
                             to apply to the map images
            pin_memory (bool): pinning memory from cpu to gpu if using gpu for inference
            num_workers (int): number of worker used for proceesing the images by the dataloader.

        Returns:
            torch.utils.data.DataLoader: a dataloader for the map image set.

        """
        pass

    @abstractmethod
    def ground_truth(self, partition: str, gt_type: str) -> np.ndarray:
        """
          This function return sthe relevant ground truth matrix given the partition
          and ground truth type. The return matrix "GT" is of type bool where
          GT[i, j] is true when query image i was taken from the place depicted by map image
          j

        Args:
            partition (str): determines which partition the datasets map images to return.
                             must bet either "train", "val", "test", or "all"
            gt_type (str): either "hard" or "soft". see https://arxiv.org/abs/2303.03281 for an
                           explanation of soft and hard ground truth in Visual Place Recognition.

        Returns:
            np.ndarray: A matrix GT of boolean type where GT[i, j] is true when
            query image i was taken from the place depicted by map image. Otherwise it is false.
        """
        pass
