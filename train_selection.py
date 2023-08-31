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


METHODS = ["netvlad", "hybridnet"]

torch.set_float32_matmul_precision("medium")

WEIGHTS_NAME = "gsvcities_netvlad_hybridnet_oneright"
TRAIN_DATASET_PATH = "/home/oliver/Documents/github/VisualLoc/SelectionData/gsvcities_combinedrecall@1_oneright_train.csv"
VAL_DATASET_PATH = "/home/oliver/Documents/github/VisualLoc/SelectionData/gsvcities_combinedrecall@1_oneright_val.csv"
TEST_DATASET_PATH = "/home/oliver/Documents/github/VisualLoc/SelectionData/gsvcities_combinedrecall@1_oneright_test.csv"
TARGET_SIZE = 2
BATCH_SIZE = 196
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
        query_img_path = record[2]
        ref_img_path = record[3]
        targ = [self.df.iloc[idx][method] for method in METHODS]
        targ = np.argmax(targ)

        if self.preprocess:
            query_img = self.preprocess(
                Image.fromarray(np.array(Image.open(query_img_path))[:, :, :3])
            )
            map_img = self.preprocess(
                Image.fromarray(np.array(Image.open(ref_img_path))[:, :, :3])
            )
            img = torch.vstack((query_img, map_img))
        else:
            raise NotImplementedError
        return img, torch.tensor(targ)


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


class ResNet18(pl.LightningModule):
    def __init__(self, output_dim):
        super(ResNet18, self).__init__()

        # Load the pretrained ResNet18 model
        # self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model = resnet18(pretrained=False)

        # Modify the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, output_dim)

        # expand the first input layer to take 6 channels
        old_conv = self.model.conv1
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias,
        )
        # copy the weights to the first and second 3 channels
        new_conv.weight.data[:, :3, :, :] = old_conv.weight.data
        new_conv.weight.data[:, 3:, :, :] = old_conv.weight.data
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()
        self.model.conv1 = new_conv

        # loss function
        # self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

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
        y_hat = F.sigmoid(self(x))
        loss = self.loss_fn(y_hat, y)
        y_pred = y_hat > 0.5
        y_true = y > 0.5
        acc = torch.sum(y_pred & y_true) / (y_pred.shape[0] * y_pred.shape[1])
        self.log("test_acc", acc)
        self.log("test_loss", loss)
        return torch.sum(y_pred & y_true) / (y_pred.shape[0] * y_pred.shape[1])

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005
        )
        return optimizer


if __name__ == "__main__":
    datamodule = SelectionDataModule(batch_size=BATCH_SIZE)

    logger = TensorBoardLogger("tb_logs", name="resnet18_model")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="SelectionNetworkCheckpoints",
        filename=WEIGHTS_NAME + "_resnet18-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    # Initialize the model
    model = ResNet18(output_dim=TARGET_SIZE)
    model.train()

    # Define the trainer
    trainer = pl.Trainer(max_epochs=100, logger=logger, callbacks=[checkpoint_callback])

    # Train the model
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
