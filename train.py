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
from PlaceRec.utils import ImageDataset, get_method
from torchvision.transforms import v2
import yaml
import argparse
import pytorch_lightning as pl
from torch import optim
import torch.nn as nn
from PlaceRec.Training import TripletModule, TripletDataModule

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser()


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
    type=int,
    default=config["train"]["negs_num_per_query"],
    help="Choose the number of processing the threads for the dataloader",
)


parser.add_argument(
    "--margin",
    type=float,
    default=config["train"]["margin"],
    help="Choose the number of processing the threads for the dataloader",
)


parser.add_argument(
    "--train_batch_size",
    type=int,
    default=config["train"]["train_batch_size"],
    help="Choose the number of processing the threads for the dataloader",
)

parser.add_argument(
    "--learning_rate",
    type=float,
    default=config["train"]["learning_rate"],
    help="Choose the number of processing the threads for the dataloader",
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=config["train"]["num_workers"],
    help="Choose the number of processing the threads for the dataloader",
)

parser.add_argument(
    "--patience",
    type=int,
    default=config["train"]["patience"],
    help="Choose the early stopping patience number"
    )

parser.add_argument(
    "--seed",
    type=int,
    default=config["train"]["seed"],
    help="seed the training run"
)

parser.add_argument(
    "--training_type",
    type=str,
    default=config["train"]["training_type"],
    help="seed the training run",
)

parser.add_argument(
    "--method",
    type=str,
    default=config["train"]["method"],
    help="seed the training run",
)

args = parser.parse_args()

pl.seed_everything(args.seed)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    method = get_method(args.method)
    model = method.model
    tripletdatamodule = TripletDataModule(args, method.preprocess)
    tripletmodule = TripletModule(args, model, tripletdatamodule)

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="gpu")
    trainer.fit(tripletmodule, datamodule=tripletdatamodule)
    print("recall@5: ",tripletmodule.recallAtN(1))
    #


