import pytorch_lightning as pl
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.models import resnet18
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models import ResNet18_Weights
import torch.nn as nn

torch.set_float32_matmul_precision("medium")

WEIGHTS_NAME = "combined_largesets_recall@1"
TRAIN_DATASET_PATH = "/home/oliver/Documents/github/VisualLoc/SelectionData/nordlands_spring_nordlands_winter_nordlands_summer_recall@1_train.csv"
VAL_DATASET_PATH = "/home/oliver/Documents/github/VisualLoc/SelectionData/nordlands_spring_nordlands_winter_nordlands_summer_recall@1_val.csv"
TEST_DATASET_PATH = "/home/oliver/Documents/github/VisualLoc/SelectionData/nordlands_spring_nordlands_winter_nordlands_summer_recall@1_test.csv"
TARGET_SIZE = 3
BATCH_SIZE = 128
NUM_WORKERS = 16


select_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class SelectDataset(Dataset):
    def __init__(self, dataset_path=None, preprocess=None):
        if dataset_path == None:
            raise Exception("dataset path is not given")

        self.df = pd.read_csv(dataset_path)
        self.preprocess = preprocess

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        record = self.df.iloc[idx].to_numpy()
        query_img_path = record[1]
        ref_img_path = record[2]
        targ = record[3:]
        if self.preprocess:
            query_img = self.preprocess(
                Image.fromarray(np.array(Image.open(query_img_path))[:, :, :3])
            )
            map_img = self.preprocess(
                Image.fromarray(np.array(Image.open(ref_img_path))[:, :, :3])
            )
            img = (query_img, map_img)
        else:
            raise NotImplementedError
        return img, torch.Tensor(targ.astype(np.float32))


class SelectionDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 128):
        super().__init__()
        self.batch_size = batch_size

        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None

    def setup(self, stage=None):
        # Load datasets
        self.train_ds = SelectDataset(
            dataset_path=TRAIN_DATASET_PATH, preprocess=select_transform
        )
        self.val_ds = SelectDataset(
            dataset_path=VAL_DATASET_PATH, preprocess=select_transform
        )
        self.test_ds = SelectDataset(
            dataset_path=TEST_DATASET_PATH, preprocess=select_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=NUM_WORKERS
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )


class ParallelResnet18(pl.LightningModule):
    def __init__(self, output_dim):
        super(ParallelResnet18, self).__init__()

        # Base ResNet18 model
        base_resnet1 = resnet18(pretrained=True)
        base_resnet2 = resnet18(pretrained=True)

        # Extract layers up to the desired layer (e.g., third layer)
        self.features1 = nn.Sequential(*list(base_resnet1.children())[:4])
        self.features2 = nn.Sequential(*list(base_resnet2.children())[:4])

        # Additional layers after concatenation
        self.additional_layers = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, output_dim),
        )

        # loss function
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        img1, img2 = x
        output1 = self.features1(img1)
        output2 = self.features2(img2)

        # Concatenate outputs along the channel dimension
        combined = torch.cat((output1, output2), dim=1)

        # Pass through additional layers
        output = self.additional_layers(combined)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Log training loss to TensorBoard
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Log validation loss to TensorBoard
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        y_pred = y_hat > 0.5
        y_true = y > 0.5
        acc = torch.sum(y_pred & y_true) / (y_pred.shape[0] * y_pred.shape[1])
        self.log("test_acc", acc)
        self.log("test_loss", loss)
        return torch.sum(y_pred & y_true) / (y_pred.shape[0] * y_pred.shape[1])

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    datamodule = SelectionDataModule(batch_size=BATCH_SIZE)

    logger = TensorBoardLogger("tb_logs", name="parallel_model_low")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="SelectionNetworkCheckpoints",
        filename=WEIGHTS_NAME + "_ParallelResnet18_low-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    # Initialize the model
    model = ParallelResnet18(output_dim=TARGET_SIZE)

    # Define the trainer
    trainer = pl.Trainer(max_epochs=200, logger=logger, callbacks=[checkpoint_callback])

    # Train the model
    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule)
