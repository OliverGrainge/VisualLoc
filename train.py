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
from parsers import train_arguments
from PlaceRec.Training import TripletDataModule, TripletModule
from PlaceRec.utils import ImageDataset, get_config, get_method
from PlaceRec.Training import DistillationDataModule, DistillationModule

config = get_config()
args = train_arguments()


if __name__ == "__main__":
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")
  

    ################################# Contrastive Training ################################
    if args.training_type == "contrastive":
        method = get_method(args.method, pretrained=False)
        model = method.model.to(args.device)
        # Early Stopping CallBack
        early_stop_callback = EarlyStopping(
            monitor="recallat" + str(args.recall_values[1]),
            min_delta=0.00,
            patience=args.patience,
            verbose=False,
            mode="max",
        )

        # Checkpointing the Model
        checkpoint_callback = ModelCheckpoint(
            monitor="recallat" + str(args.recall_values[1]),
            filename=join(
                os.getcwd(), "PlaceRec/Training/checkpoints/", args.training_type, method.name, method.name + "-{epoch:02d}-{recallat1:.2f}"
            ),
            save_top_k=1,
            verbose=False,
            mode="max",
        )

        logger = WandbLogger(project=method.name, log_model="all")  # need to login to a WandB account
        logger.experiment.config.update(config["train"])  # Log the training configuration

        # Build the Datamodule
        tripletdatamodule = TripletDataModule(args, method.preprocess)
        tripletdatamodule.setup()
        # Build the Training Module
        tripletmodule = TripletModule(args, model, tripletdatamodule)

        # Build the Training Class
        trainer = pl.Trainer(
            check_val_every_n_epoch=args.val_check_interval,
            log_every_n_steps=20,
            max_epochs=args.max_epochs,
            accelerator= "gpu" if args.device in ["mps", "cuda"] else "cpu",
            logger=logger,
            reload_dataloaders_every_n_epochs=1,
            callbacks=[checkpoint_callback],
        )

        # Initiate Training
        trainer.fit(tripletmodule, datamodule=tripletdatamodule)


    ###################### Asymmetric Distillation Training #######################
    elif args.training_type == "asymmetric_distillation":
        student_method = get_method(args.student_method, pretrained=False)
        teacher_method = get_method(args.teacher_method, pretrained=False)
        # Early Stopping CallBack
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=args.patience,
            verbose=False,
            mode="min",
        )

        # Checkpointing the Model
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            filename=join(
                os.getcwd(), "PlaceRec/Training/checkpoints/", args.training_type, student_method.name, student_method.name + "-{epoch:02d}-{recallat1:.2f}"
            ),
            save_top_k=1,
            verbose=False,
            mode="min",
        )

        logger = WandbLogger(project="asymmetric distillation")  # need to login to a WandB account
        logger.experiment.config.update(config["train"])  # Log the training configuration

        # Build the Datamodule
        distillationdatamodule = DistillationDataModule(args, teacher_method, teacher_method.preprocess, reload=args.reload)
        # Build the Training Module
        distillationmodule = DistillationModule(args, student_method)



        # Build the Training Class
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator= "gpu" if args.device in ["mps", "cuda"] else "cpu",
            logger=logger,
            callbacks=[early_stop_callback, checkpoint_callback],
        )

        # Initiate Training
        trainer.fit(distillationmodule, datamodule=distillationdatamodule)

