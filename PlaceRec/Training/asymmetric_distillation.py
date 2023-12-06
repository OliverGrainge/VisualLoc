import argparse
import logging
import os
from glob import glob
from os.path import join

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as T
import yaml
from IPython.display import display
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.transforms import v2
from tqdm import tqdm
import random
import torchvision.transforms.functional as TF



class BaseDistillationDataset(data.Dataset):
    """
    A PyTorch Dataset for loading and processing images for model inference.

    This dataset is used for both training and testing, handling the loading of database and query images. It also computes soft and hard positives for each query image.

    Parameters:
    - args (Namespace): Arguments containing dataset configuration.
    - split (str): Indicates the dataset split to use ('train' or 'test').

    Raises:
    - FileNotFoundError: If the dataset folders do not exist.
    """

    def __init__(self, args, preprocess, split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = args.dataset_name

        self.dataset_folder = join(args.datasets_folder, args.dataset_name, "images", split)
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")

        database_folder = join(self.dataset_folder, "database")
        queries_folder = join(self.dataset_folder, "queries")

        if not os.path.exists(database_folder):
            raise FileNotFoundError(f"Folder {database_folder} does not exist")
        if not os.path.exists(queries_folder):
            raise FileNotFoundError(f"Folder {queries_folder} does not exist")

        self.database_paths = sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
        self.queries_paths = sorted(glob(join(queries_folder, "**", "*.jpg"), recursive=True))

        self.all_images_paths = list(self.database_paths) + list(self.queries_paths)
        self.all_images_num = len(self.database_paths) + list(self.queries_paths)

    def __len__(self):
        return self.all_images_num

    def __getitem__(self, idx):
        return Image.open(self.all_images_paths[idx]).convert("RGB"), idx




class RadomResizeDataset(data.Dataset):
    def __init__(self, basedistillationdataset, preprocess, min_size=224, max_size=420):
        self.base_dataset = basedistillationdataset
        self.preprocess = preprocess
    
    def __len__(self):
        return self.base_dataset.__len__()

    def __getitem__(self, idx): 
        pil_img = self.base_dataset.__getitem__(idx)
        img = self.preprocess(pil_img)
        new_size = random.randint(self.min_size, self.max_size)
        return TF.resize(img, new_size)





class TripletDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning Data Module for handling triplet datasets.

    This module prepares the dataset for training and validation, handling the necessary transformations and data loading.

    Parameters:
    - args (Namespace): Arguments containing dataset and model configuration.
    - test_preprocess (function): Transformation function for test data.
    - train_preprocess (function, optional): Transformation function for train data.
    """

    def __init__(self, args, test_preprocess, train_preprocess=None, reload):
        super().__init__()
        self.args = args
        self.test_preprocess = test_preprocess
        if train_preprocess is not None:
            self.train_preprocess = train_preprocess
        else:
            self.train_preprocess = test_preprocess

        self.pin_memory = True if self.args.device == "cuda" else False

    def setup(self, stage=None):
        # Split data into train, validate, and test sets
        self.train_dataset = BaseDistillationDataset(self.args, self.test_preprocess, self.train_preprocess, split="train")
        self.test_dataset = BaseDistillationDataset(self.args, self.test_preprocess, split="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.args.train_batch_size, num_workers=self.args.num_workers, pin_memory=self.pin_memory, shuffle=True
        )
    


