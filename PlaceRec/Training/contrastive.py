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

from PlaceRec.utils import (
    ImageIdxDataset,
    get_loss_function,
    get_method,
)


def test(args, eval_ds, model, preprocess):
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
        database_descs = np.empty((eval_ds.database_num, args.features_dim), dtype="float32")
        database_ds = ImageIdxDataset(eval_ds.database_paths, preprocess)
        database_dl = DataLoader(database_ds, num_workers=args.num_workers, batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        i = 0
        progress_bar = tqdm(total=len(database_dl), desc="Computing Test Database Descriptors")
        for inputs, indicies in database_dl:
            features = model(inputs.to(args.device)).cpu().numpy()
            database_descs[indicies.cpu().numpy(), :] = features
            progress_bar.update(1)
        progress_bar.close()

        queries_descs = np.empty((eval_ds.queries_num, args.features_dim), dtype="float32")
        queries_ds = ImageIdxDataset(eval_ds.queries_paths, preprocess)
        queries_dl = DataLoader(queries_ds, num_workers=args.num_workers, batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        progress_bar = tqdm(total=len(queries_dl), desc="Computing Test Query Descriptors")
        for inputs, indicies in queries_dl:
            features = model(inputs.to(args.device)).cpu().numpy()
            queries_descs[indicies.cpu().numpy(), :] = features
            progress_bar.update()
        progress_bar.close()

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

    _, predictions = index.search(queries_descs, max(args.recall_values))
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


class BaseTrainingDataset(data.Dataset):
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
        self.hard_positives_per_query = list(
            knn.radius_neighbors(self.queries_utms, radius=args.train_positive_dist_threshold, return_distance=False)  # 10 meters
        )
        queries_without_any_hard_positive = np.where(np.array([len(p) for p in self.hard_positives_per_query], dtype=object) == 0)[0]
        if len(queries_without_any_hard_positive) != 0:
            print(
                f"There are {len(queries_without_any_hard_positive)} queries without any positives "
                + "within the training set. They won't be considered as they're useless for training."
            )
        # Remove queries without positives
        self.hard_positives_per_query = [arr for i, arr in enumerate(self.hard_positives_per_query) if i not in queries_without_any_hard_positive]
        self.queries_paths = [arr for i, arr in enumerate(self.queries_paths) if i not in queries_without_any_hard_positive]

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
    """
    A PyTorch Dataset for generating image triplets for training.

    This dataset inherits from BaseTrainingDataset and adds functionality to create triplets consisting of an anchor, a positive, and multiple negatives.

    Parameters:
    - args (Namespace): Arguments containing dataset and mining configuration.
    - test_preprocess (function): Preprocessing function for testing images.
    - train_preprocess (function): Preprocessing function for training images.
    - split (str): Dataset split to use ('train' or 'test').

    Methods:
    - features_size(model): Returns the size of the features extracted by the model.
    - view_triplets(sample_size): Displays sample triplets.
    - mine_triplets(model): Mines triplets based on the specified mining strategy.
    """

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

    def features_size(self, model):
        img = Image.open(self.queries_paths[0]).convert("RGB")
        img = self.test_preprocess(img)
        with torch.no_grad():
            features = model(img[None, :].to(self.args.device)).detach().cpu()
            return features.size(1)

    def view_triplets(self, sample_size=5):
        if len(self.triplets) == 0:
            self.mine_triplets()

        idxs = np.random.choice(np.arange(len(self.triplets)), size=sample_size)
        for idx in idxs:
            print(idx)
            triplet = self.triplets[idx]
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(Image.open(triplet[0]))
            ax[0].set_title("Anchor")
            ax[1].imshow(Image.open(triplet[1]))
            ax[1].set_title("Positive")
            ax[2].imshow(Image.open(triplet[2]))
            ax[2].set_title("Negative")
            ax[0].axis("off")
            ax[1].axis("off")
            ax[2].axis("off")
        plt.show()

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
        hard_positives_per_query = [np.random.choice(self.hard_positives_per_query[idx]) for idx in sample_query_indexes]

        sample_negatives_per_query = [
            np.random.choice(np.setdiff1d(np.arange(self.database_num), hard_positives_per_query[i]), size=10)
            for i in range(len(hard_positives_per_query))
        ]
        sample_negatives_per_query = np.array(sample_negatives_per_query).flatten()
        sample_negatives_per_query_paths = [self.database_paths[idx] for idx in sample_negatives_per_query]

        pin_memory = True if self.args.device == "cuda" else False
        mining_ds = ImageIdxDataset(np.concatenate((sample_query_paths, sample_negatives_per_query_paths)), preprocess=self.test_preprocess)
        mining_loader = DataLoader(
            mining_ds, batch_size=self.args.infer_batch_size, num_workers=self.args.num_workers, pin_memory=pin_memory, shuffle=False
        )

        self.args.features_dim = self.features_size(model)
        all_descs = np.empty((len(sample_query_paths) + len(sample_negatives_per_query_paths), self.args.features_dim), dtype="float32")
        progress_bar = tqdm(total=len(mining_loader), desc="Mining Hard Negatives", leave=False)
        with torch.no_grad():
            for batch, indicies in mining_loader:
                features = model(batch.to(self.args.device)).detach().cpu()
                all_descs[indicies.numpy(), :] = features
                progress_bar.update(1)

        queries_desc, negatives_desc = all_descs[: len(sample_query_paths)], all_descs[len(sample_query_paths) :]

        if self.args.loss_distance == "cosine":
            faiss.normalize_L2(negatives_desc)
            index = faiss.IndexFlatIP(negatives_desc.shape[1])
            index.add(negatives_desc)
            faiss.normalize_L2(queries_desc)
            _, similarities = index.search(queries_desc, self.args.neg_num_per_query)
        elif self.args.loss_distance == "l2":
            index = faiss.IndexFlatL2(negatives_desc.shape[1])
            index.add(negatives_desc)
            _, similarities = index.search(queries_desc, self.args.neg_num_per_query)
        else:
            raise Exception(self.args.loss_distance + "distance partial mining not implemented")

        self.triplets = [
            [self.queries_paths[sample_query_indexes[i]], self.database_paths[hard_positives_per_query[i]]]
            + [self.database_paths[similarities[i, j]] for j in range(self.args.neg_num_per_query)]
            for i in range(self.args.cache_refresh_rate)
        ]
        progress_bar.close()

    def random_mine(self, model):
        query_idxs = np.random.choice(np.arange(self.queries_num), size=self.args.cache_refresh_rate)
        positive_idxs = np.array([np.random.choice(self.hard_positives_per_query[v]) for v in query_idxs])
        neg_idxs = np.array(
            [np.random.choice(np.setdiff1d(np.arange(self.database_num), pos_idx), size=self.args.neg_num_per_query) for pos_idx in positive_idxs]
        )
        self.triplets = []
        for i in range(len(query_idxs)):
            anchor_path = self.queries_paths[query_idxs[i]]
            positive_path = self.database_paths[positive_idxs[i]]
            negatives_paths = [self.database_paths[idx] for idx in neg_idxs[i]]
            self.triplets.append([anchor_path, positive_path] + negatives_paths)



################################################### Data Module ###################################################


class TripletDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning Data Module for handling triplet datasets.

    This module prepares the dataset for training and validation, handling the necessary transformations and data loading.

    Parameters:
    - args (Namespace): Arguments containing dataset and model configuration.
    - test_preprocess (function): Transformation function for test data.
    - train_preprocess (function, optional): Transformation function for train data.
    """

    def __init__(self, args, test_preprocess, train_preprocess=None):
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
        self.train_dataset = TripletDataset(self.args, self.test_preprocess, self.train_preprocess, split="train")
        self.test_dataset = TripletDataset(self.args, self.test_preprocess, self.train_preprocess, split="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.args.train_batch_size, num_workers=self.args.num_workers, pin_memory=self.pin_memory, shuffle=True
        )
    
    def val_dataloader(self):
        queries_ds = ImageIdxDataset(self.test_dataset.queries_paths, self.test_preprocess)
        queries_dl = DataLoader(queries_ds, num_workers=self.args.num_workers, batch_size=self.args.infer_batch_size, pin_memory=(self.args.device == "cuda"))
        database_ds = ImageIdxDataset(self.test_dataset.database_paths, self.test_preprocess)
        database_dl = DataLoader(database_ds, num_workers=self.args.num_workers, batch_size=self.args.infer_batch_size, pin_memory=(self.args.device == "cuda"))
        return queries_dl, database_dl
    







################################################### Training Module ###################################################


class TripletModule(pl.LightningModule):
    """
    A PyTorch Lightning Module for training with triplet loss.

    This module encapsulates the model training process using triplet loss. It handles the training steps, validation, and testing, as well as optimizer configuration.

    Parameters:
    - args (Namespace): Arguments containing model and training configuration.
    - model (nn.Module): The neural network model to train.
    - datamodule (TripletDataModule): The data module containing training and validation data.

    Methods:
    - features_size(): Returns the size of the features extracted by the model.
    - on_train_start(): Prepares the dataset for training.
    - on_train_epoch_end(): Updates the mining of triplets at the end of each training epoch.
    - forward(x): Forward pass through the model.
    - training_step(batch, batch_idx): Defines the training step.
    - on_validation_epoch_end(outputs): Handles actions at the end of a validation epoch.
    - on_test_epoch_end(batch, batch_idx, dataloader_idx): Handles actions at the end of a test epoch.
    - configure_optimizers(): Configures the model optimizers.
    """

    def __init__(self, args, model, datamodule):
        super().__init__()
        self.args = args
        self.model = model
        self.datamodule = datamodule
        self.loss_fn = get_loss_function(args)
        self.save_hyperparameters(ignore=["model"])
        self.features_dim = self.features_size()

    def features_size(self):
        img = Image.open(self.datamodule.train_dataset.queries_paths[0]).convert("RGB")
        img = self.datamodule.test_preprocess(img)
        with torch.no_grad():
            features = self.model(img[None, :].to(self.args.device)).detach().cpu()
            return features.size(1)

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
    
    def on_validation_epoch_start(self):
        self.validation_queries_descs = np.empty((self.datamodule.test_dataset.queries_num, self.features_dim), dtype=np.float32)
        self.validation_database_descs = np.empty((self.datamodule.test_dataset.database_num, self.features_dim), dtype=np.float32)
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        batch, indicies = batch
        features = self.model(batch)

        if dataloader_idx == 0:
            self.validation_queries_descs[indicies.cpu().numpy(), :] = features.detach().cpu().numpy()
        elif dataloader_idx == 1:
            self.validation_database_descs[indicies.cpu().numpy(), :] = features.detach().cpu().numpy()
        

    def on_validation_epoch_end(self):  
        if self.args.loss_distance == "cosine":
            faiss.normalize_L2(self.validation_database_descs)
            index = faiss.IndexFlatIP(self.features_dim)
            index.add(self.validation_database_descs)
            faiss.normalize_L2(self.validation_queries_descs)
        elif self.args.loss_distance == "l2":
            index = faiss.IndexFlatL2(self.features_dim)
            index.add(self.validation_database_descs)

        distances, predictions = index.search(self.validation_queries_descs, max(self.args.recall_values))
        #### For each query, check if the predictions are correct
        positives_per_query = self.datamodule.test_dataset.soft_positives_per_query
        # args.recall_values by default is [1, 5, 10, 20]
        recalls = np.zeros(len(self.args.recall_values))
        for query_index, pred in enumerate(predictions):
            for i, n in enumerate(self.args.recall_values):
                if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break

        # Divide by the number of queries*100, so the recalls are in percentages
        recalls = recalls / self.datamodule.test_dataset.queries_num * 100
        recalls_str = ", ".join([f"R@{val}: {rec:.4f}" for val, rec in zip(self.args.recall_values, recalls)])


        print("")
        print("")
        print(recalls_str)
        print("")
        print("")
        for i, recall in enumerate(recalls):
            self.log("recallat" + str(self.args.recall_values[i]), recall, on_epoch=True)
        return recalls[1]
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return optimizer
