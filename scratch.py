from train.dataloaders.train.GSVCitiesDataset import GSVCitiesDataset
from train import GSVCitiesDataModule
from PlaceRec.utils import get_method
from PlaceRec import utils
import numpy as np
from torch.utils.data import DataLoader 
from os.path import join 
import os
from PlaceRec.utils import get_config
from parsers import train_arguments

config = get_config()
args = train_arguments()


METHODS = ["cosplace"]


TRAIN_CITIES = [
    "Bangkok",
    "BuenosAires",
    "LosAngeles",
    "MexicoCity",
    "OSL",  # refers to Oslo
    "Rome",
    "Barcelona",
    "Chicago",
    "Madrid",
    "Miami",
    "Phoenix",
    "TRT",  # refers to Toronto
    "Boston",
    "Lisbon",
    "Medellin",
    "Minneapolis",
    "PRG",  # refers to Prague
    "WashingtonDC",
    "Brussels",
    "London",
    "Melbourne",
    "Osaka",
    "PRS",  # refers to Paris
]

DEBUG_CITIES = ["TRT"]

ds = GSVCitiesDataset(cities=DEBUG_CITIES)
dm = GSVCitiesDataModule(args)

out = ds.__getitem__(0)

dl = dm.train_dataloader()

for batch in dl:
    print(len(batch))
    print(batch[0].shape, batch[1])