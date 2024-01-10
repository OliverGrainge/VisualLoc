import argparse
import logging
import os
import random
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
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.transforms.functional as F
import yaml
from IPython.display import display
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.transforms import v2
from tqdm import tqdm
import random
from PlaceRec.utils import ImageIdxDataset

file_directory = os.path.dirname(os.path.abspath(__file__))


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

    def __init__(self, args, split="train"):
        super().__init__()
        self.args = args
        self.dataset_directory = args.datasets_directory

        self.image_folders = glob(self.dataset_directory + "/gsv_cities/Images/*")
        self.all_images_paths = []
        for folder in self.image_folders:
            self.all_images_paths += glob(join(folder + "/*.jpg"))
        if split == "train":
            self.all_images_paths = self.all_images_paths[1000:100000]
        else: 
            self.all_images_paths = self.all_images_paths[:1000]
        self.all_images_num = len(self.all_images_paths)

    def __len__(self):
        return self.all_images_num

    def __getitem__(self, idx):
        return Image.open(self.all_images_paths[idx]).convert("RGB"), idx


class TeacherDataset(data.Dataset):
    def __init__(self, basedistillationdataset, preprocess):
        self.base_dataset = basedistillationdataset
        self.preprocess = preprocess

    def __len__(self):
        return self.base_dataset.__len__()

    def __getitem__(self, idx):
        pil_img, _ = self.base_dataset.__getitem__(idx)
        img = self.preprocess(pil_img)
        return img, idx

class UniformRandomResizeDataset(data.Dataset):
    def __init__(self, basedistillationdataset, cache, preprocess, mult_range=(0.8, 1.2), teacher_size=(480, 640)):
        self.base_dataset = basedistillationdataset
        self.cache = cache
        self.preprocess = preprocess
        self.teacher_size = teacher_size
        self.mult_range = mult_range

    def __len__(self):
        return self.base_dataset.__len__()

    def __getitem__(self, idx):
        pil_img, _ = self.base_dataset.__getitem__(idx)
        img = self.preprocess(pil_img)
        return img, self.cache[idx], idx

    def collate_fn(self, batch):
        random_mult = self.mult_range[0] + (np.random.rand() * (self.mult_range[1] - self.mult_range[0]))
        res = (int(random_mult * self.teacher_size[0]), int(random_mult * self.teacher_size[1]))
        imgs = [F.resize(item[0], res, antialias=True) for item in batch]
        features = [torch.tensor(item[1]) for item in batch]
        idxs = [item[2] for item in batch]
        imgs = torch.stack(imgs, dim=0)
        features = torch.vstack(features)
        idxs = torch.tensor(idxs)
        return imgs, features, idxs


class SingleResolutionDataset(data.Dataset):
    def __init__(self, basedistillationdataset, cache, preprocess, size=(320, 320)):
        self.base_dataset = basedistillationdataset
        self.cache = cache
        self.preprocess = preprocess
        self.resize = transforms.Resize(size, antialias=True)
    
    def __len__(self):
        return self.base_dataset.__len__()
    
    def __getitem__(self, idx):
        pil_img, _ = self.base_dataset.__getitem__(idx)
        img = self.preprocess(pil_img)
        img = self.resize(img)
        return img, self.cache[idx], idx
    
    def collate_fn(self, batch):
        imgs = [item[0] for item in batch]
        features = [torch.tensor(item[1]) for item in batch]
        idxs = [item[2] for item in batch]
        imgs = torch.stack(imgs, dim=0)
        features = torch.vstack(features)
        idxs = torch.tensor(idxs)
        return imgs, features, idxs



class DistillationDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning Data Module for handling triplet datasets.

    This module prepares the dataset for training and validation, handling the necessary transformations and data loading.

    Parameters:
    - args (Namespace): Arguments containing dataset and model configuration.
    - test_preprocess (function): Transformation function for test data.
    - train_preprocess (function, optional): Transformation function for train data.
    """

    def __init__(self, args, teacher_method, test_preprocess, train_preprocess=None, reload=False):
        super().__init__()
        self.args = args
        self.test_preprocess = test_preprocess
        self.teacher_method = teacher_method
        self.teacher_model = teacher_method.model
        self.teacher_model.to(self.args.device)
        self.teacher_model.eval()
        self.reload = reload

        if train_preprocess is not None:
            self.train_preprocess = train_preprocess
        else:
            self.train_preprocess = test_preprocess

        self.pin_memory = True if self.args.device == "cuda" else False

    def setup(self, stage=None):
        # Split data into train, validate, and test sets
        if self.reload:
            try:
                # try and load the cache
                self.train_cache = torch.load(
                    os.path.join(file_directory, "utils", self.teacher_method.name, "train_features.pt")
                )
                self.val_cache = torch.load(
                    os.path.join(file_directory, "utils", self.teacher_method.name, "val_features.pt")
                )
            except:
                self.compute_teacher_features()
        else:
            self.compute_teacher_features()

        self.base_train_dataset = BaseDistillationDataset(self.args, split="train")
        self.base_val_dataset = BaseDistillationDataset(self.args, split="test")

        if self.args.distillation_type == "uniform_random_resize":
            self.student_train_dataset = UniformRandomResizeDataset(
                self.base_train_dataset, self.train_cache, preprocess=self.train_preprocess, mult_range = (self.args.min_mult, self.args.max_mult), teacher_size=(480, 640)
            )
            self.student_val_dataset = UniformRandomResizeDataset(
                self.base_val_dataset, self.val_cache, preprocess=self.test_preprocess, mult_range = (self.args.min_mult, self.args.max_mult), teacher_size=(480, 640)
            )
        elif self.args.distillation_type == "single_resolution":
            self.student_train_dataset = SingleResolutionDataset(self.base_train_dataset, self.train_cache, preprocess = self.train_preprocess, size=self.args.size)
            self.student_val_dataset = SingleResolutionDataset(self.base_val_dataset, self.val_cache, preprocess=self.test_preprocess, size=self.args.size)

    def compute_teacher_features(self):
        train_dataset = TeacherDataset(BaseDistillationDataset(self.args, "train"), preprocess=self.test_preprocess)
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=(self.args.device == "cuda")
        )
        self.train_cache = self.compute_cache(train_dataloader, "Computing Training Teacher Features: ")
        del train_dataloader
        val_dataset = TeacherDataset(BaseDistillationDataset(self.args, "test"), preprocess=self.test_preprocess)
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=(self.args.device == "cuda")
        )
        self.val_cache = self.compute_cache(val_dataloader, "Computing Validation Teacher Features: ")
        del val_dataloader

        if not os.path.exists(os.path.join(file_directory, "utils", self.teacher_method.name)):
            os.makedirs(os.path.join(file_directory, "utils", self.teacher_method.name))

        torch.save(self.train_cache, os.path.join(file_directory, "utils", self.teacher_method.name, "train_features.pt"))
        torch.save(self.val_cache, os.path.join(file_directory, "utils", self.teacher_method.name, "val_features.pt"))

    def compute_cache(self, dataloader, desc):
        cache = np.empty((dataloader.dataset.__len__(), self.teacher_method.features_dim), dtype=np.float32)
        with torch.no_grad():
            for batch, indicies in tqdm(dataloader, desc=desc):
                features = self.teacher_model(batch.to(self.args.device)).detach().cpu().numpy()
                cache[indicies, :] = features
        return cache

    def train_dataloader(self):
        return DataLoader(
            self.student_train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.student_train_dataset.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.student_val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=self.student_val_dataset.collate_fn,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.student_val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=self.student_val_dataset.collate_fn,
            pin_memory=self.pin_memory,
        )


class DistillationModule(pl.LightningModule):
    def __init__(self, args, student_method):
        super(DistillationModule, self).__init__()
        self.args = args
        self.student_method = student_method
        self.model = student_method.model
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs, target_features, idx = batch
        features = self.model(imgs)
        loss = self.loss_fn(features, target_features)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, target_features, idx = batch
        features = self.model(imgs)
        loss = self.loss_fn(features, target_features)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch):
        imgs, target_features, idx = batch
        features = self.model(imgs)
        loss = self.loss_fn(features, target_features)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return optimizer