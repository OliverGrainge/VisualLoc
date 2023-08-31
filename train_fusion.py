from PlaceRec.utils import get_method
from PlaceRec.utils import get_dataset

# Import necessary libraries
import pytorch_lightning as pl
from torchvision import models
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pytorch_lightning import Trainer, LightningDataModule
import torch.nn.functional as F
import wandb
from pytorch_lightning.loggers import WandbLogger

METHODS = ["amosnet", "hybridnet"]
DATASETS = ["nordlands_summer", "nordlands_winter", "nordlands_spring"]
PARTITION = "train"
BATCH_SIZE = 64


transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class FusionDataset(Dataset):
    def __init__(
        self, methods=METHODS, datasets=DATASETS, transform=None, partition="test"
    ):
        self.transform = transform

        self.similarity = []
        self.ground_truth = []
        self.query_images = []

        for dataset in tqdm(datasets, desc="Loading dataset"):
            sim = []
            ds = get_dataset(dataset)
            ground_truth = ds.ground_truth(partition, gt_type="hard")
            self.ground_truth.append(ground_truth)
            self.query_images += ds.query_partition(partition).tolist()
            for method in methods:
                method = get_method(method)
                method.load_descriptors(ds.name, partition=partition)
                similarity = method.similarity_matrix(
                    method.query_desc, method.map_desc
                )
                del method

                assert similarity.shape == ground_truth.shape
                sim.append(similarity)
            self.similarity.append(sim)

        self.boundaries = []
        cumulative_cols = 0
        for matrix in self.ground_truth:
            _, cols = matrix.shape
            lower_bound = cumulative_cols
            upper_bound = cumulative_cols + cols - 1
            self.boundaries.append((lower_bound, upper_bound))
            cumulative_cols += cols

    def __len__(self):
        return np.sum([gt.shape[1] for gt in self.ground_truth])

    def __getitem__(self, idx):
        for i, (lower_bound, upper_bound) in enumerate(self.boundaries):
            if lower_bound <= idx <= upper_bound:
                gt = self.ground_truth[i][:, idx - lower_bound]
                sim_vects = [sim[:, idx - lower_bound] for sim in self.similarity[i]]
                break

        if self.transform:
            img = self.transform(
                Image.fromarray(np.array(Image.open(self.query_images[idx]))[:, :, :3])
            )

        return (
            img,
            torch.tensor(np.array(sim_vects)).float(),
            torch.tensor(gt).float(),
        )


class FusionDataModule(LightningDataModule):
    def __init__(self, batch_size: int = BATCH_SIZE, transform=transform):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):
        # Transform and split data
        if stage == "fit" or stage is None:
            self.train_dataset = FusionDataset(
                partition="train", transform=self.transform
            )
            self.val_dataset = FusionDataset(partition="val", transform=self.transform)

        if stage == "test" or stage is None:
            self.test_dataset = FusionDataset(
                partition="test", transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


# Define the Lightning Module
class ResNet18(pl.LightningModule):
    def __init__(self, output_dim):
        super(ResNet18, self).__init__()

        # Load pretrained ResNet18 model
        self.model = models.resnet18(pretrained=True)

        # Modify the fully connected layer to have arbitrary output dimension
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, output_dim)

        # Define the loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, sim, gt = batch
        weights = self(img)
        fuse_sim = self.fuse_similarity(sim, weights)
        loss = self.loss_fn(fuse_sim, gt.argmax(1))
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def fuse_similarity(self, sim_vects, fuse_weights):
        weights = F.softmax(fuse_weights, dim=1)
        weighted_sim = sim_vects * weights[:, :, None]
        fused_sim = torch.sum(weighted_sim, dim=1)
        return fused_sim

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        img, sim, gt = batch
        weights = self(img)
        fuse_sim = self.fuse_similarity(sim, weights)

        loss = self.loss_fn(fuse_sim, gt.argmax(1))
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        img, sim, gt = batch
        weights = self(img)
        fuse_sim = self.fuse_similarity(sim, weights)
        loss = self.loss_fn(fuse_sim, gt.argmax(1))
        self.log("test_loss", loss)
        return {"test_loss": loss}


if __name__ == "__main__":
    wandb.init(project="FusionVPR")
    wandblogger = WandbLogger(project="FusionVPR")
    trainer = Trainer(max_epochs=5, accelerator="gpu", logger=wandblogger)

    model = ResNet18(output_dim=len(METHODS))
    dm = FusionDataModule()
    trainer.fit(model, dm)
