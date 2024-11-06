import math
from argparse import Namespace
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from onnxruntime.quantization import CalibrationDataReader
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class ImageIdxDataset(Dataset):
    def __init__(self, img_paths, preprocess=None):
        self.img_paths = img_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.preprocess is not None:
            img = np.array(Image.open(self.img_paths[idx]).convert("RGB"))[:, :, :3]
            img = Image.fromarray(img)
            img = self.preprocess(img)
            return idx, img

        img = np.array(
            Image.open(self.img_paths[idx]).convert("RGB").resize((224, 224))
        )[:, :, :3]
        return idx, img


def get_dataset(name: str = None):
    module_name = "PlaceRec.Datasets"
    dataset_module = __import__(module_name, fromlist=[name])
    dataset_class = getattr(dataset_module, name)
    return dataset_class()


def get_method(name: str = None, pretrained: bool = True):
    module_name = "PlaceRec.Methods"
    method_module = __import__(module_name, fromlist=[name])
    method_class = getattr(method_module, name)
    return method_class(pretrained=pretrained)


def cosine_distance(x1, x2):
    cosine_sim = nn.CosineSimilarity(dim=0)(x1, x2)
    return 1 - cosine_sim


def get_loss_function(args):
    if args.loss_distance == "l2":
        return nn.TripletMarginLoss(args.margin, p=2, reduction="sum")
    elif args.loss_distance == "cosine":
        return nn.TripletMarginWithDistanceLoss(
            distance_function=cosine_distance, margin=args.margin
        )


def get_config():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class QuantizationDataReader(CalibrationDataReader):
    def __init__(self, dataloader, max_inputs: int = 500):
        """
        Initializes the QuantizationDataReader with a PyTorch DataLoader.

        Args:
            dataloader (torch.utils.data.DataLoader): PyTorch DataLoader providing calibration data.
        """
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
        self.max_batches = math.ceil(dataloader.batch_size / max_inputs)
        self.batch_counter = 0

    def get_next(self):
        """
        Provides the next batch of inputs for ONNX Runtime calibration.

        Returns:
            Dict[str, np.ndarray] or None: A dictionary where the keys match the input names of the ONNX model
                                           and the values are the input data as numpy arrays. Returns None when
                                           the data is exhausted.
        """
        try:
            if self.batch_counter >= self.max_batches:
                return None  # End of data
            data = next(self.data_iter)
            inputs = data[1]
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.cpu().numpy()
            self.batch_counter += 1
            return {"input": inputs}

        except StopIteration:
            return None
