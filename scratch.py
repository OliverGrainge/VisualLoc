import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer


from PlaceRec.Methods import AmosNet 
from PlaceRec.Datasets import StLucia_small

method = AmosNet()
ds = StLucia_small()
ql = ds.query_images_loader("test", preprocess=method.preprocess)
ml = ds.map_images_loader("test", preprocess=method.preprocess)

method.compute_map_desc(dataloader=ml)
method.compute_query_desc(dataloader=ql)

method.save_descriptors("testing_dataset")
print(method.map_desc["map_descriptors"][0])

mdesc = method.map_desc
method.map_desc = 0

print(method.map_desc)
method.load_descriptors("testing_dataset")

print(method.map_desc["map_descriptors"][0])

print((mdesc["map_descriptors"] == method.map_desc["map_descriptors"]))

print("============")

import numpy as np
print(np.max(np.abs(mdesc["map_descriptors"] - method.map_desc["map_descriptors"])))

print(np.allclose(mdesc["map_descriptors"], method.map_desc["map_descriptors"], atol=0.000001))

assert (mdesc["map_descriptors"] == method.map_desc["map_descriptors"]).all()




"""
# MNIST Data Module
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        # Download only once
        MNIST('', train=True, download=True)
        MNIST('', train=False, download=True)

    def setup(self, stage=None):
        # Split and transform the dataset
        if stage == 'fit' or stage is None:
            mnist_full = MNIST('', train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST('', train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def on_epoch_end(self):
        # Call on_epoch_end of DataModule
        #self.trainer.datamodule.on_epoch_end()
        # Other actions for the end of epoch in the model
        print("Epoch ended in Model")
        raise Exception

# Define the model
class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(28 * 28, 128)
        self.layer2 = torch.nn.Linear(128, 256)
        self.layer3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return F.log_softmax(self.layer3(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def on_train_epoch_end(self):
        # Call on_epoch_end of DataModule
        #self.trainer.datamodule.on_epoch_end()
        # Other actions for the end of epoch in the model
        train_dataset = self.trainer.train_dataloader.dataset
        print(f"Epoch ended in Model {train_dataset.__len__()}")


# Use the DataModule
batch_size = 64
mnist_dm = MNISTDataModule(batch_size=batch_size)

# Model
model = MNISTModel()

# Train
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, mnist_dm)

"""