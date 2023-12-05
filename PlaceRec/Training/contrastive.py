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
from .geobench_contrastive import GeoBaseDataset



import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset



def test(args, eval_ds, model, test_method="hard_resize", pca=None):
    """Compute features of the given dataset and compute the recalls."""
    
    assert test_method in ["hard_resize", "single_query", "central_crop", "five_crops",
                           "nearest_crop", "maj_voting"], f"test_method can't be {test_method}"
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        
    
        all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")

        for inputs, indices in tqdm(database_dataloader, ncols=100):
            features = model(inputs.to(args.device))
            features = features.cpu().numpy()
            if pca is not None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features
        
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        eval_ds.test_method = test_method
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            features = model(inputs.to(args.device))
            features = features.cpu().numpy()            
            all_features[indices.numpy(), :] = features
    
    queries_features = all_features[eval_ds.database_num:]
    database_features = all_features[:eval_ds.database_num]
    
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del database_features, all_features
    
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))
    

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
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
    
    def view_triplets(self, sample_size=5):
        if len(self.triplets) == 0:
            self.mine_triplets()

        idxs = np.random.choice(np.arange(len(self.triplets)), size=sample_size)
        for idx in idxs:
            print(idx)
            triplet = self.triplets[idx]

            fig, ax = plt.subplots(3)
           
            ax[0].imshow(Image.open(triplet[0]))
            ax[0].set_title("Anchor")
            ax[1].imshow(Image.open(triplet[1]))
            ax[1].set_title("Positive")
            ax[2].imshow(Image.open(triplet[2]))
            ax[2].set_title("Negative")
            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
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
        mining_ds = ImageDataset(np.concatenate((sample_query_paths, sample_negatives_per_query_paths)), preprocess=self.test_preprocess)
        mining_loader = DataLoader(mining_ds, batch_size=self.args.infer_batch_size, num_workers=self.args.num_workers, pin_memory=pin_memory, shuffle=False)

        all_desc = []
        progress_bar = tqdm(total=len(mining_loader), desc="Mining Hard Negatives", leave=False)
        with torch.no_grad():
            for batch in mining_loader:
                all_desc.append(model(batch.to(self.args.device)).detach().cpu())
                progress_bar.update(1)

        all_desc = torch.vstack(all_desc).numpy().astype(np.float32)
        queries_desc, negatives_desc = all_desc[:len(sample_query_paths)], all_desc[len(sample_query_paths):]

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


        self.triplets = [
            [self.queries_paths[sample_query_indexes[i]], self.database_paths[hard_positives_per_query[i]]]
            + [self.database_paths[similarities[i, j]] for j in range(self.args.negs_num_per_query)]
            for i in range(self.args.cache_refresh_rate)
        ]
        progress_bar.close()

    def random_mine(self, model):
        query_idxs = np.random.choice(np.arange(self.queries_num), size=self.args.cache_refresh_rate)
        # sample a random
        # raise Exception("Need to Remove the Queries in the Dataset without Any Positives")

        positive_idxs = np.array([np.random.choice(self.hard_positives_per_query[v]) for v in query_idxs])
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
        query_dl = DataLoader(query_ds, batch_size=self.args.infer_batch_size, num_workers=self.args.num_workers, pin_memory=self.pin_memory, shuffle=False)
        database_dl = DataLoader(database_ds, batch_size=self.args.infer_batch_size, num_workers=self.args.num_workers, pin_memory=self.pin_memory, shuffle=False)
        return [query_dl, database_dl]


class TripletModule(pl.LightningModule):
    def __init__(self, args, model, datamodule):
        super().__init__()
        self.args = args
        self.model = model
        self.datamodule = datamodule
        self.loss_fn = get_loss_function(args)
        self.save_hyperparameters()


    def features_size(self):
        dl, _ = self.datamodule.recall_dataloader()
        for batch in dl:
            features = self.model(batch.to(self.args.device)).detach().cpu()
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

    def on_validation_epoch_end(self, outputs):
        self.args.features_dim = self.features_size()
        eval_ds = GeoBaseDataset(self.args, self.args.datasets_folder, self.args.dataset_name, "test")
        recalls, recalls_str = test(self.args, eval_ds, self.model, test_method="hard_resize")
        self.log("val_loss", recalls[1])
        return recalls[1]

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


    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return optimizer
