import logging
import os
from glob import glob
from os.path import join

import faiss
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
from PlaceRec.utils import ImageDataset
from torchvision.transforms import v2
import yaml
import argparse

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--resize",
    type=tuple,
    default=config["train"]["resize"],
    help="Choose the number of processing the threads for the dataloader",
)



parser.add_argument(
    "--dataset_name",
    type=str,
    default=config["train"]["dataset_name"],
    help="Choose the number of processing the threads for the dataloader",
)

parser.add_argument(
    "--datasets_folder",
    type=str,
    default=config["train"]["datasets_folder"],
    help="Choose the number of processing the threads for the dataloader",
)


parser.add_argument(
    "--val_positive_dist_threshold",
    type=int,
    default=config["train"]["val_positive_dist_threshold"],
    help="Choose the number of processing the threads for the dataloader",
)


parser.add_argument(
    "--mining",
    type=str,
    default=config["train"]["mining"],
    help="Choose the number of processing the threads for the dataloader",
)



parser.add_argument(
    "--neg_samples_num",
    type=int,
    default=config["train"]["neg_samples_num"],
    help="Choose the number of processing the threads for the dataloader",
)

parser.add_argument(
    "--infer_batch_size",
    type=int,
    default=config["train"]["infer_batch_size"],
    help="Choose the number of processing the threads for the dataloader",
)


parser.add_argument(
    "--cache_refresh_rate",
    type=int,
    default=config["train"]["cache_refresh_rate"],
    help="Choose the number of processing the threads for the dataloader",
)

parser.add_argument(
    "--device",
    type=str,
    default=config["train"]["device"],
    help="Choose the number of processing the threads for the dataloader",
)

parser.add_argument(
    "--negs_num_per_query",
    type=str,
    default=config["train"]["negs_num_per_query"],
    help="Choose the number of processing the threads for the dataloader",
)


args = parser.parse_args()

class BaseDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache)."""

    def __init__(self, args, split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = args.dataset_name

        self.test_preprocess = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(args.resize),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.dataset_folder = join(args.datasets_folder, args.dataset_name, "images", split)
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")

        self.resize = args.resize

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



class TripletDataset(BaseDataset): 
    def __init__(self, args, test_preprocess, train_preprocess, split="train"):
        super().__init__(args, split=split)

        self.test_preprocess = test_preprocess
        self.train_preprocess = train_preprocess

        self.triplets = []
        self.query_desc = []
        self.map_desc = []

    def __len__(self):
        return len(self.triplet_cache)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        anchor = self.train_preprocess(Image.open(triplet[0]).convert("RGB"))
        positive = self.train_preprocess(Image.open(triplet[1]).convert("RGB"))
        negatives = [self.train_preprocess(Image.open(pth).convert("RGB")) for pth in triplet[2:]]
        return anchor, positive, negatives

    def mine_triplets(self, model):
        if self.args.mining == "partial":
            self.cache = self.partial_mine(model)
        elif self.args.mining == "random":
            self.cache = self.random_mine(model)
        elif self.args.mining == "full":
            raise NotImplementedError

    def partial_mine(self, model):
        model.eval()
        model.to(self.args.device)
        sample_query_indexes = np.random.choice(np.arange(self.queries_num), size=self.args.cache_refresh_rate)
        sample_query_paths = [self.queries_paths[idx] for idx in sample_query_indexes]

        soft_positives_per_query = [np.random.choice(self.soft_positives_per_query[idx]) for idx in sample_query_indexes]
        sample_negatives_per_query = [np.random.choice(np.setdiff1d(np.arange(self.database_num), soft_positives_per_query[i]), size=10) for i in range(len(soft_positives_per_query))]
        sample_negatives_per_query = np.array(sample_negatives_per_query).flatten()
        sample_negatives_per_query_paths = [self.database_paths[idx] for idx in sample_negatives_per_query]
        
        negatives_dataset = ImageDataset(sample_negatives_per_query_paths, preprocess=self.test_preprocess)
        negatives_dataloader = DataLoader(negatives_dataset, batch_size=self.args.infer_batch_size)
        queries_dataset = ImageDataset(sample_query_paths, preprocess=self.test_preprocess)
        queries_dataloader = DataLoader(queries_dataset, batch_size=self.args.infer_batch_size)

        with torch.no_grad():
            negatives_desc = torch.vstack([model(batch.to(self.args.device)).detach().cpu() for batch in negatives_dataloader]).numpy().astype(np.float32)
            queries_desc = torch.vstack([model(batch.to(self.args.device)).detach().cpu() for batch in queries_dataloader]).numpy().astype(np.float32)
        
        faiss.normalize_L2(negatives_desc)
        index = faiss.IndexFlatIP(negatives_desc.shape[1])
        index.add(negatives_desc)
        faiss.normalize_L2(queries_desc)
        _, similarities = index.search(queries_desc, self.args.neg_samples_num)

        hard_negative_sample_idxs = np.array([np.random.choice(np.arange(similarities.shape[1]), size=self.args.negs_num_per_query, replace=False) for _ in range(similarities.shape[0])])
        hard_negatives = np.array([similarities[np.arange(similarities.shape[0]), hard_neg] for hard_neg in hard_negative_sample_idxs.transpose()]).transpose()

        self.triplets = [[self.queries_paths[sample_query_indexes[i]],
                          self.database_paths[soft_positives_per_query[i]]] + 
                          [self.database_paths[hard_negatives[i, j]] for j in range(self.args.negs_num_per_query)] for i in range(self.args.cache_refresh_rate)]



    def random_mine(self, model):
        # sample a random subset of queries from the queries
        sample_query_indexes = np.random.choice(np.arange(self.queries_num), size=self.args.cache_refresh_rate)
        # sample a random 
        soft_positives_per_query = [np.random.choice(self.soft_positives_per_query[idx]) for idx in sample_query_indexes]
        negatives = np.array([np.random.choice(np.setdiff1d(np.arange(self.database_num), soft_positives_per_query[i]), size=self.args.negs_num_per_query) for i in range(len(soft_positives_per_query))])
        self.triplets = [[self.queries_paths[sample_query_indexes[i]],
                          self.database_paths[soft_positives_per_query[i]]] + 
                          [self.database_paths[negatives[i][j]] for j in range(self.args.negs_num_per_query)] for i in range(self.args.cache_refresh_rate)]






from PlaceRec.Methods import AmosNet
method = AmosNet()
model = method.model
ds = TripletDataset(args, method.preprocess, method.preprocess, split="test")
ds.random_mine(model)

anchor, positive, negative = ds.__getitem__(2)

print(type(negative))
print(len(negative))

