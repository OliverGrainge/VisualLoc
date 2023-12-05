import argparse
import os
from glob import glob
from os.path import join

import faiss
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
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.transforms import v2
from tqdm import tqdm
import matplotlib.pyplot as plt

from PlaceRec.utils import ImageDataset, get_loss_function, get_method


package_directory = os.path.dirname(os.path.abspath(__file__))


class BaseTrainingDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache)."""

    def __init__(self, args, split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = args.dataset_name

        self.dataset_folder = join(args.datasets_folder, args.dataset_name, "images", split)
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")

        #### Read paths and UTM coordinates for all images.
        database_folder = join(self.dataset_folder, "database")
        queries_folder = join(self.dataset_folder, "queries")

        if not os.path.exists(database_folder):
            raise FileNotFoundError(f"Folder {database_folder} does not exist")
        if not os.path.exists(queries_folder):
            raise FileNotFoundError(f"Folder {queries_folder} does not exist")

        self.database_paths = sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
        self.queries_paths = sorted(glob(join(queries_folder, "**", "*.jpg"), recursive=True))

        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)
        self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(float)

        # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.soft_positives_per_query = knn.radius_neighbors(
            self.queries_utms,
            radius=args.val_positive_dist_threshold,
            return_distance=False,
        )

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.hard_positives_per_query = list(knn.radius_neighbors(self.queries_utms,
                                             radius=args.train_positive_dist_threshold,  # 10 meters
                                             return_distance=False))


        queries_without_any_hard_positive = np.where(np.array([len(p) for p in self.hard_positives_per_query], dtype=object) == 0)[0]
        if len(queries_without_any_hard_positive) != 0:
            print(f"There are {len(queries_without_any_hard_positive)} queries without any positives " +
                         "within the training set. They won't be considered as they're useless for training.")
            
        # Remove queries without positives
        self.hard_positives_per_query = [
            arr
            for i, arr in enumerate(self.hard_positives_per_query)
            if i not in queries_without_any_hard_positive
        ]


        self.queries_paths = [
            arr
            for i, arr in enumerate(self.queries_paths)
            if i not in queries_without_any_hard_positive
        ]

        self.all_images_paths = list(self.database_paths) + list(self.queries_paths)
        self.queries_paths = list(self.queries_paths)
        self.database_paths = list(self.database_paths)
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)

    def __len__(self):
        return self.queries_num

    def __getitem__(self, idx):
        return Image.open(self.all_images_paths[idx]).convert("RGB"), idx



class DistillationDataset(data.Dataset):
    def __init__(self, cache, preprocess):
        super().__init__()
        self.cache = cache
        self.preprocess = preprocess

    def __len__(self):
        return self.cache.size(0)
    

    
    





class DistilationDataModule(pl.LightningDataModule):
    def __init__(self, args, teacher, test_transform, train_transform=None, reload=False):
        super().__init__()
        self.args = args
        self.reload = reload
        self.test_transform = test_transform
        if train_transform is not None:
            self.train_transform = train_transform
        else:
            self.train_transform = test_transform

        self.teacher = teacher
        self.teacher.eval()
        self.teacher.to(args.device)

        self.pin_memory = True if self.args.device == "cuda" else False

    def setup(self, stage=None):
        # Split data into train, validate, and test sets
        self.train_dataset = BaseTrainingDataset(self.args, self.test_transform, self.train_transform, split="train")
        self.test_dataset = BaseTrainingDataset(self.args, self.test_transform, self.train_transform, split="test")
        if self.reload == True: 
            try:
                self.train_cache = torch.load(package_directory + "/utils/train_teacher_features.pth")
                self.val_cache = torch.load(package_directory + "/utils/val_teacher_features.pth")
            except:
                print("Failed To Load Features")
                self.train_cache = self.compute_teacher_cache("train")
                self.val_cache = self.compute_teacher_cache("val")
        else: 
            self.train_cache = self.compute_teacher_cache("train")
            self.val_cache = self.compute_teacher_cache("val")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.args.train_batch_size, num_workers=self.args.num_workers, pin_memory=self.pin_memory, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.train_batch_size, num_workers=self.args.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.train_batch_size, num_workers=self.args.num_workers, pin_memory=self.pin_memory)

    def compute_teacher_cache(self, partition):   
        if partition == "train":         
            dl = self.train_dataloader()
        elif partition == "val":
            dl = self.val_dataloader()
        with torch.no_grad():
            cache = []
            for batch in tqdm(dl, desc=f'Extracting teacher {partition} features'):
                cache.append(self.teacher(batch.to(self.args.device)).detach().cpu())
            cache = torch.vstack(cache)
        return cache