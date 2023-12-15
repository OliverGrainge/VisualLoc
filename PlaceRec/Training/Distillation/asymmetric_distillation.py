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

from PlaceRec.utils import ImageIdxDataset

file_directory = os.path.dirname(os.path.abspath(__file__))


def test(args, eval_ds, model, preprocess, database_descs):
    """
    Evaluate the model on a given dataset.

    This function computes descriptors for the database and query images, and then uses these descriptors to perform image retrieval.
    The retrieval performance is measured using recall values at different cut-off points.

    Parameters:
    - args (Namespace): Arguments containing model, dataset, and evaluation configuration.
    - eval_ds (Dataset): The evaluation dataset containing paths to database and query images.
    - model (nn.Module): Trained model to compute image descriptors.
    - preprocess (function): Preprocessing function to apply to images before passing them to the model.

    Returns:
    - tuple: A tuple containing the recall values and a formatted string of recall values.
    """

    model = model.eval()

    with torch.no_grad():
        queries_descs = np.empty((eval_ds.queries_num, args.features_dim), dtype="float32")
        queries_ds = ImageIdxDataset(eval_ds.queries_paths, preprocess)
        queries_dl = DataLoader(queries_ds, num_workers=args.num_workers, batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        for inputs, indicies in tqdm(queries_dl, desc="Computing Test Query Descriptors"):
            features = model(inputs.to(args.device)).cpu().numpy()
            queries_descs[indicies.numpy(), :] = features

        if args.loss_distance == "cosine":
            faiss.normalize_L2(database_descs)
            index = faiss.IndexFlatIP(args.features_dim)
            index.add(database_descs)
            faiss.normalize_L2(queries_descs)
            del database_descs
        elif args.loss_distance == "l2":
            index = faiss.IndexFlatL2(args.features_dim)
            index.add(database_descs)
            del database_descs

    distances, predictions = index.search(queries_descs, max(args.recall_values))
    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.soft_positives_per_query
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    return recalls, recalls_str


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
    def __init__(self, basedistillationdataset, cache, preprocess, min_size=224, max_size=420):
        self.base_dataset = basedistillationdataset
        self.cache = cache
        self.preprocess = preprocess
        self.min_size = min_size
        self.max_size = max_size

    def __len__(self):
        return self.base_dataset.__len__()

    def __getitem__(self, idx):
        pil_img, _ = self.base_dataset.__getitem__(idx)
        img = self.preprocess(pil_img)
        return img, self.cache[idx], idx

    def collate_fn(self, batch):
        res = torch.randint(low=self.min_size, high=self.max_size, size=(1,))
        imgs = [F.resize(item[0], (res, res), antialias=True) for item in batch]
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
                    os.path.join(file_directory, "utils", self.args.dataset_name, self.teacher_method.name, "train_features.pt")
                )
                self.val_cache = torch.load(
                    os.path.join(file_directory, "utils", self.args.dataset_name, self.teacher_method.name, "val_features.pt")
                )
            except:
                self.compute_teacher_features()
        else:
            self.compute_teacher_features()

        self.base_train_dataset = BaseDistillationDataset(self.args, split="train")
        self.base_val_dataset = BaseDistillationDataset(self.args, split="test")

        if self.args.distillation_type == "uniform_random_resize":
            self.student_train_dataset = UniformRandomResizeDataset(
                self.base_train_dataset, self.train_cache, preprocess=self.train_preprocess, min_size=self.args.min_size, max_size=self.args.max_size
            )
            self.student_val_dataset = UniformRandomResizeDataset(
                self.base_val_dataset, self.val_cache, preprocess=self.test_preprocess, min_size=self.args.min_size, max_size=self.args.max_size
            )

    def compute_teacher_features(self):
        train_dataset = TeacherDataset(BaseDistillationDataset(self.args, "train"), preprocess=self.test_preprocess)
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.args.infer_batch_size, num_workers=self.args.num_workers, pin_memory=(self.args.device == "cuda")
        )
        self.train_cache = self.compute_cache(train_dataloader, "Computing Training Teacher Features: ")
        del train_dataloader
        val_dataset = TeacherDataset(BaseDistillationDataset(self.args, "test"), preprocess=self.test_preprocess)
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.args.infer_batch_size, num_workers=self.args.num_workers, pin_memory=(self.args.device == "cuda")
        )
        self.val_cache = self.compute_cache(val_dataloader, "Computing Validation Teacher Features: ")
        del val_dataloader

        if not os.path.exists(os.path.join(file_directory, "utils", self.args.dataset_name, self.teacher_method.name)):
            os.makedirs(os.path.join(file_directory, "utils", self.args.dataset_name, self.teacher_method.name))

        torch.save(self.train_cache, os.path.join(file_directory, "utils", self.args.dataset_name, self.teacher_method.name, "train_features.pt"))
        torch.save(self.val_cache, os.path.join(file_directory, "utils", self.args.dataset_name, self.teacher_method.name, "val_features.pt"))

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
            batch_size=self.args.train_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.student_train_dataset.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.student_val_dataset,
            batch_size=self.args.infer_batch_size,
            num_workers=self.args.num_workers,
            collate_fn=self.student_val_dataset.collate_fn,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.student_val_dataset,
            batch_size=self.args.infer_batch_size,
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


def update_preprocess_size(all_preprocess, new_size):
    new_preprocess = []
    for i, transform in enumerate(all_preprocess.transforms):
        if isinstance(transform, transforms.Resize):
            new_preprocess.append(transforms.Resize(new_size, antialias=True))
        else:
            new_preprocess.append(transform)
    return transforms.Compose(new_preprocess)


def recallatn(args, model, test_dataset, test_preprocess):
    model.eval()
    queries_descs = np.empty((test_dataset.queries_num, args.features_dim), dtype=np.float32)
    database_descs = np.empty((test_dataset.database_num, args.features_dim), dtype=np.float32)

    queries_dl = DataLoader(
        ImageIdxDataset(test_dataset.queries_paths, test_preprocess),
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    database_dl = DataLoader(
        ImageIdxDataset(test_dataset.database_paths, test_preprocess),
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    with torch.no_grad():
        for batch, idxs in tqdm(queries_dl, desc="Validation Query Features"):
            features = model(batch).detach().cpu().numpy()
            queries_descs[idxs.detach().cpu().numpy(), :] = features

        for batch, idxs in tqdm(database_dl, desc="Validation Database Features"):
            features = model(batch).detach().cpu().numpy()
            database_descs[idxs.detach().cpu().numpy(), :] = features

    index = faiss.IndexFlatL2(args.features_dim)
    index.add(database_descs)
    del database_descs

    _, predictions = index.search(queries_descs, max(args.recall_values))

    pos_per_query = test_dataset.soft_positives_per_query
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], pos_per_query[query_index])):
                recalls[i:] += 1
                break
    recalls = recalls / test_dataset.queries_num * 100
    return recalls


def recallvsresolution(args, model, test_preprocess, n_points=10):
    base_preprocess = test_preprocess
    sizes = np.linspace(args.min_size, args.max_size, n_points).astype(int)
    all_recalls = []
    for size in sizes:
        test_preprocess = update_preprocess_size(base_preprocess, size)
        test_dataset = contrastive.BaseDataset(
            args, test_preprocess, datasets_folder=args.datasets_folder, dataset_name=args.dataset_name, split="test"
        )
        recalls = recallatn(args, model, test_dataset, test_preprocess)
        all_recalls.append(recalls)
    return sizes, np.vstack(all_recalls)
