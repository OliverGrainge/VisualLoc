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
from parsers import train_arguments
from PlaceRec.utils import get_config

import wandb
from PlaceRec.Training import TripletDataModule, TripletModule
from PlaceRec.utils import ImageDataset, get_method

config = get_config()
args = train_arguments()


if __name__ == "__main__":
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")
    method = get_method(args.method, pretrained=True)
    model = method.model.to(args.device)


    if args.training_type == "contrastive":
        early_stop_callback = EarlyStopping(
            monitor="val_loss",  # Metric to monitor
            min_delta=0.00,  # Minimum change to qualify as an improvement
            patience=args.patience,  # Number of epochs with no improvement after which training will be stopped
            verbose=False,  # Whether to output a message when early stopping is triggered
            mode="max",  # Mode - 'min' for minimizing the metric, 'max' for maximizing
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="Recall@" + str(args.recall_values[1]),
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
            val_check_interval=args.val_check_interval,
            max_epochs=args.max_epochs,
            accelerator=args.device,
            logger=logger,
            callbacks=[early_stop_callback, checkpoint_callback],
        )

        trainer.fit(tripletmodule, datamodule=tripletdatamodule)
