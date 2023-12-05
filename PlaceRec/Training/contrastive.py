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

from PlaceRec.utils import ImageDataset, get_loss_function, get_method


class BaseTrainingDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache)."""

    def __init__(self, args, split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = args.dataset_name

        self.test_preprocess = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((480, 640)),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

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

        self.all_images_paths = list(self.database_paths) + list(self.queries_paths)
        self.queries_paths = list(self.queries_paths)
        self.database_paths = list(self.database_paths)
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)

    def __len__(self):
        return self.queries_num

    def __getitem__(self, idx):
        return Image.open(self.queries_paths[idx]).convert("RGB"), idx


class TripletDataset(BaseTrainingDataset):
    def __init__(self, args, test_preprocess, train_preprocess, split="train"):
        super().__init__(args, split=split)

        self.test_preprocess = test_preprocess
        self.train_preprocess = train_preprocess

        self.triplets = []
        self.query_desc = []
        self.map_desc = []

        self.random_mine(None)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        anchor = self.train_preprocess(Image.open(triplet[0]).convert("RGB"))
        positive = self.train_preprocess(Image.open(triplet[1]).convert("RGB"))
        negatives = [self.train_preprocess(Image.open(pth).convert("RGB")) for pth in triplet[2:]]

        return anchor, positive, negatives

    def mine_triplets(self, model):
        if self.args.mining == "partial":
            self.partial_mine(model)
        elif self.args.mining == "random":
            self.random_mine(model)
        elif self.args.mining == "full":
            raise NotImplementedError

    def partial_mine(self, model):
        model.eval()
        model.to(self.args.device)
        sample_query_indexes = np.random.choice(np.arange(self.queries_num), size=self.args.cache_refresh_rate, replace=False)
        sample_query_paths = [self.queries_paths[idx] for idx in sample_query_indexes]

        soft_positives_per_query = [np.random.choice(self.soft_positives_per_query[idx]) for idx in sample_query_indexes]

        sample_negatives_per_query = [
            np.random.choice(np.setdiff1d(np.arange(self.database_num), soft_positives_per_query[i]), size=10)
            for i in range(len(soft_positives_per_query))
        ]
        sample_negatives_per_query = np.array(sample_negatives_per_query).flatten()
        sample_negatives_per_query_paths = [self.database_paths[idx] for idx in sample_negatives_per_query]

        pin_memory = True if self.args.device == "cuda" else False

        negatives_dataset = ImageDataset(sample_negatives_per_query_paths, preprocess=self.test_preprocess)
        negatives_dataloader = DataLoader(
            negatives_dataset, batch_size=self.args.infer_batch_size, num_workers=self.args.num_workers, pin_memory=pin_memory
        )
        queries_dataset = ImageDataset(sample_query_paths, preprocess=self.test_preprocess)
        queries_dataloader = DataLoader(
            queries_dataset, batch_size=self.args.infer_batch_size, num_workers=self.args.num_workers, pin_memory=pin_memory
        )

        progress_bar = tqdm(total=len(negatives_dataloader) + len(queries_dataloader), desc="Mining Hard Negatives", leave=False)
        with torch.no_grad():
            negatives_desc_acc = []
            queries_desc_acc = []
            for batch in negatives_dataloader:
                negatives_desc_acc.append(model(batch.to(self.args.device)).detach().cpu())
                progress_bar.update(1)

            negatives_desc = torch.vstack(negatives_desc_acc).numpy().astype(np.float32)

            for batch in queries_dataloader:
                queries_desc_acc.append(model(batch.to(self.args.device)).detach().cpu())
                progress_bar.update(1)

            queries_desc = torch.vstack(queries_desc_acc).numpy().astype(np.float32)

        if self.args.loss_distance == "cosine":
            faiss.normalize_L2(negatives_desc)
            index = faiss.IndexFlatIP(negatives_desc.shape[1])
            index.add(negatives_desc)
            faiss.normalize_L2(queries_desc)
            _, similarities = index.search(queries_desc, self.args.neg_samples_num)
        elif self.args.loss_distance == "l2":
            index = faiss.IndexFlatL2(negatives_desc.shape[1])
            index.add(negatives_desc)
            _, similarities = index.search(queries_desc, self.args.neg_samples_num)
        else:
            raise Exception(self.args.loss_distance + "distance partial mining not implemented")

        hard_negative_sample_idxs = np.array(
            [
                np.random.choice(np.arange(similarities.shape[1]), size=self.args.negs_num_per_query, replace=False)
                for _ in range(similarities.shape[0])
            ]
        )
        hard_negatives = np.array(
            [similarities[np.arange(similarities.shape[0]), hard_neg] for hard_neg in hard_negative_sample_idxs.transpose()]
        ).transpose()

        self.triplets = [
            [self.queries_paths[sample_query_indexes[i]], self.database_paths[soft_positives_per_query[i]]]
            + [self.database_paths[hard_negatives[i, j]] for j in range(self.args.negs_num_per_query)]
            for i in range(self.args.cache_refresh_rate)
        ]
        progress_bar.close()

    def random_mine(self, model):
        query_idxs = np.random.choice(np.arange(self.queries_num), size=self.args.cache_refresh_rate)
        # sample a random
        # raise Exception("Need to Remove the Queries in the Dataset without Any Positives")

        positive_idxs = np.array([np.random.choice(self.soft_positives_per_query[v]) for v in query_idxs])
        neg_idxs = np.array(
            [np.random.choice(np.setdiff1d(np.arange(self.database_num), pos_idx), size=self.args.negs_num_per_query) for pos_idx in positive_idxs]
        )
        self.triplets = []
        for i in range(len(query_idxs)):
            anchor_path = self.queries_paths[query_idxs[i]]
            positive_path = self.database_paths[positive_idxs[i]]
            negatives_paths = [self.database_paths[idx] for idx in neg_idxs[i]]
            self.triplets.append([anchor_path, positive_path] + negatives_paths)


class TripletDataModule(pl.LightningDataModule):
    def __init__(self, args, test_transform, train_transform=None):
        super().__init__()
        self.args = args
        self.test_transform = test_transform
        if train_transform is not None:
            self.train_transform = train_transform
        else:
            self.train_transform = test_transform

        self.pin_memory = True if self.args.device == "cuda" else False

    def setup(self, stage=None):
        # Split data into train, validate, and test sets
        self.train_dataset = TripletDataset(self.args, self.test_transform, self.train_transform, split="train")
        self.test_dataset = TripletDataset(self.args, self.test_transform, self.train_transform, split="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.args.train_batch_size, num_workers=self.args.num_workers, pin_memory=self.pin_memory, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.train_batch_size, num_workers=self.args.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.train_batch_size, num_workers=self.args.num_workers, pin_memory=self.pin_memory)

    def recall_dataloader(self):
        query_ds = ImageDataset(self.test_dataset.queries_paths, self.test_transform)
        database_ds = ImageDataset(self.test_dataset.database_paths, self.test_transform)
        query_dl = DataLoader(query_ds, batch_size=self.args.infer_batch_size, num_workers=self.args.num_workers, pin_memory=self.pin_memory)
        database_dl = DataLoader(database_ds, batch_size=self.args.infer_batch_size, num_workers=self.args.num_workers, pin_memory=self.pin_memory)
        return [query_dl, database_dl]


class TripletModule(pl.LightningModule):
    def __init__(self, args, model, datamodule):
        super().__init__()
        self.args = args
        self.model = model
        self.datamodule = datamodule
        self.loss_fn = get_loss_function(args)
        self.save_hyperparameters()

    def on_train_start(self):
        self.datamodule.train_dataset.mine_triplets(self.model)
        self.datamodule.test_dataset.mine_triplets(self.model)

    def on_train_epoch_end(self):
        self.datamodule.train_dataset.mine_triplets(self.model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        anchor, positive, negatives = batch

        all_images = torch.vstack([anchor] + [positive] + negatives)
        all_desc = self.model(all_images)
        anchor_desc = all_desc[0]
        positive_desc = all_desc[1]
        negatives_desc = all_desc[2:]

        loss = 0
        for negative in negatives_desc:
            loss += self.loss_fn(anchor_desc, positive_desc, negative)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        anchor, positive, negatives = batch
        all_images = torch.vstack([anchor] + [positive] + negatives)
        all_desc = self.model(all_images)
        anchor_desc = all_desc[0]
        positive_desc = all_desc[1]
        negatives_desc = all_desc[2:]

        loss = 0
        for negative in negatives_desc:
            loss += self.loss_fn(anchor_desc, positive_desc, negative)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        anchor, positive, negatives = batch
        all_images = torch.vstack([anchor] + [positive] + negatives)
        all_desc = self.model(all_images)
        anchor_desc = all_desc[0]
        positive_desc = all_desc[1]
        negatives_desc = all_desc[2:]

        loss = 0
        for negative in negatives_desc:
            loss += self.loss_fn(anchor_desc, positive_desc, negative)
        self.log("test_loss", loss)
        return loss

    def recallAtN(self, N=5):
        # self.model.eval().to(self.args.device)
        query_loader, database_loader = self.datamodule.recall_dataloader()
        with torch.no_grad():
            database_desc_acc = []
            query_desc_acc = []
            progress_bar = tqdm(total=len(query_loader) + len(database_loader), desc="Computing Recall@N Descriptors", leave=False)
            for batch in query_loader:
                query_desc_acc.append(self.model(batch).detach().cpu())
                progress_bar.update(1)

            query_descs = torch.vstack(query_desc_acc).numpy().astype(np.float32)

            for batch in database_loader:
                database_desc_acc.append(self.model(batch).detach().cpu())
                progress_bar.update(1)

            database_descs = torch.vstack(database_desc_acc).numpy().astype(np.float32)

        if self.args.loss_distance == "cosine":
            index = faiss.IndexFlatIP(query_descs.shape[1])
            faiss.normalize_L2(query_descs)
            faiss.normalize_L2(database_descs)
            index.add(database_descs)
        elif self.args.loss_distance == "l2":
            index = faiss.IndexFlatL2(query_descs.shape[1])
            index.add(database_descs)
        else:
            raise Exception(self.args.loss_distance + "training is not implemented")

        del database_descs

        distances, predictions = index.search(query_descs, N)

        recall_count = 0
        for row1, row2 in zip(predictions, self.datamodule.test_dataset.soft_positives_per_query):
            if np.any(np.isin(row1, row2)):
                recall_count += 1

        return recall_count / predictions.shape[0]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
