import logging
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
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
from PIL import Image
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from sklearn.neighbors import NearestNeighbors
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.transforms import v2
from tqdm import tqdm

import wandb
from PlaceRec.Training import TripletDataModule, TripletModule
from PlaceRec.utils import ImageDataset, get_method

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser()

parser.add_argument("--recall_values", type=int, default=[1, 5, 10, 20], nargs="+", help="Recalls to be computed, such as R@5.")

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
    "--train_positive_dist_threshold",
    type=int,
    default=config["train"]["train_positive_dist_threshold"],
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

parser.add_argument("--patience", type=int, default=config["train"]["patience"], help="Choose the early stopping patience number")

parser.add_argument("--seed", type=int, default=config["train"]["seed"], help="seed the training run")

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

parser.add_argument(
    "--resize",
    type=int,
    default=[480, 640],
    # default=[240, 320],
    nargs=2,
    help="Resizing shape for images (HxW).",
)

parser.add_argument("--efficient_ram_testing", action="store_true", help="_")

parser.add_argument(
    "--test_method",
    type=str,
    default="hard_resize",
    choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
    help="This includes pre/post-processing methods and prediction refinement",
)


parser.add_argument("--max_epochs", type=int, default=config["train"]["max_epochs"])

parser.add_argument("--loss_distance", type=str, default=config["train"]["loss_distance"])

args = parser.parse_args()

# pl.seed_everything(args.seed)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    method = get_method(args.method, pretrained=False)
    model = method.model.to(args.device)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",  # Metric to monitor
        min_delta=0.00,  # Minimum change to qualify as an improvement
        patience=args.patience,  # Number of epochs with no improvement after which training will be stopped
        verbose=False,  # Whether to output a message when early stopping is triggered
        mode="min",  # Mode - 'min' for minimizing the metric, 'max' for maximizing
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename=join(os.getcwd(), "PlaceRec/Training/checkpoints/" + method.name + "/" + method.name + "-{epoch:02d}-{val_loss:.2f}"),
        save_top_k=1,
        verbose=False,
        mode="min",
    )

    logger = WandbLogger(project=method.name)
    logger.experiment.config.update(config["train"])

    tripletdatamodule = TripletDataModule(args, method.preprocess)
    tripletdatamodule.setup(None)
    tripletmodule = TripletModule(args, model, tripletdatamodule)

    trainer = pl.Trainer(
        val_check_interval=5,
        max_epochs=args.max_epochs,
        accelerator=args.device,
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    trainer.fit(tripletmodule, datamodule=tripletdatamodule)
    #trainer.test(tripletmodule, datamodule=tripletdatamodule)
