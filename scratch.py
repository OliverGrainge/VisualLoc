import torch
import torch.nn as nn
import torchvision.models as models
import torch_pruning as tp
from PlaceRec import Methods
import torch_pruning as tp
from PlaceRec.utils import get_dataset
import pickle

datasets = ["CrossSeason", "Pitts30k"]


def get_datasets_size():
    dataset_sizes = {}
    for dataset_name in datasets:
        ds = get_dataset(dataset_name)
        dataset_sizes[dataset_name] = len(ds.map_paths)

    with open("dataset_sizes.pkl", "wb") as f:
        pickle.dump(dataset_sizes, f)
