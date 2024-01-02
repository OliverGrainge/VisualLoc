import logging
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
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
from PIL import Image
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from sklearn.neighbors import NearestNeighbors
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.transforms import v2
from tqdm import tqdm
from PlaceRec.utils import get_training_logger

import wandb
from parsers import train_arguments
from PlaceRec.Training.Contrastive import GSVCitiesDataModule, VPRModel
from PlaceRec.Training.Distillation import (
    DistillationDataModule,
    DistillationModule,
)
from PlaceRec.utils import ImageDataset, get_config, get_method


def features_size(args, model, preprocess):
    image = torch.rand(244, 244, 3) * 255
    image = image.numpy().astype(np.uint8)
    image = Image.fromarray(image)
    image = preprocess(image)
    out = model(image[None, :].to(args.device))
    return out.shape[1]


def main(args, config):
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")  # Initiate Training
    ################################# Contrastive Training ################################
    if args.training_type == "contrastive":
        from PlaceRec.Training.Contrastive import GSVCitiesDataModule, VPRModel
        method = get_method(args.method, pretrained=False)
        model = method.model.to(args.device)

        args.features_dim = features_size(args, model, method.preprocess)
        args = train_arguments()

        datamodule = GSVCitiesDataModule(args)
        model = VPRModel(args, model)
        
        # Checkpointing the Model
        checkpoint_callback = ModelCheckpoint(
            monitor='pitts30k_val/R1',
            filename=join(
                os.getcwd(),
                "PlaceRec/Training/Contrastive/checkpoints/",
                method.name,
                f'{method.name}' + '_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]'),
                auto_insert_metric_name=False,
                save_top_k=1,
                verbose=False,
                mode="max",
            )
            
        logger = get_training_logger(config, project_name="Contrastive")
        

        trainer = pl.Trainer(
            accelerator='gpu', devices=[0],
            default_root_dir=f'PlaceRec/Training/Contrastive/Logs/{method.name}', # Tensorflow can be used to viz 
            num_sanity_val_steps=0, # runs N validation steps before stating training
            precision=16, # we use half precision to reduce  memory usage (and 2x speed on RTX)
            max_epochs=30,
            check_val_every_n_epoch=1, # run validation every epoch
            callbacks=[checkpoint_callback],# we run the checkpointing callback (you can add more)
            logger=logger,
            reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
            log_every_n_steps=20,
            limit_train_batches=10,
            #fast_dev_run=True # comment if you want to start training the network and saving checkpoints
        )
        trainer.fit(model=model, datamodule=datamodule)
        
    ###################### Distillation Training #######################
    elif args.training_type == "distillation":
        student_method = get_method(args.student_method, pretrained=False)
        teacher_method = get_method(args.teacher_method, pretrained=True)
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
                os.getcwd(),
                "PlaceRec/Distillation/Training/checkpoints/",
                args.training_type,
                student_method.name,
                student_method.name + "-{epoch:02d}-{recallat1:.2f}",
            ),
            save_top_k=1,
            verbose=False,
            mode="min",
        )

        logger = get_training_logger(config, project="Distillation")

        # Build the Datamodule
        distillationdatamodule = DistillationDataModule(args, teacher_method, teacher_method.preprocess, reload=args.reload)
        # Build the Training Module
        distillationmodule = DistillationModule(args, student_method)

        # Build the Training Class
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator="gpu" if args.device in ["mps", "cuda"] else "cpu",
            logger=logger,
            callbacks=[early_stop_callback, checkpoint_callback],
        )

        trainer.fit(distillationmodule, datamodule=distillationdatamodule)
        #recalls, resolutions = recallvsresolution(args, teacher_method.model, test_preprocess=student_method.preprocess, n_points=4)

        for col in range(recalls.shape[1]):
            rec = recalls[:, col]
            plt.plot(rec, resolutions, label=f"Recall@{args.recall_values[col]}")

        plt.plot(recalls, resolutions)
        plt.xlabel("Recall@N")
        plt.ylabel("Image Resolution")
        plt.show()


if __name__ == "__main__":
    config = get_config()
    args = train_arguments()
    main(args, config)
