import os
import pickle
import random
from glob import glob
from typing import Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from ..utils import s3_bucket_download
from .base_method import BaseModelWrapper

package_directory = os.path.dirname(os.path.abspath(__file__))


class CalcModel(nn.Module):
    def __init__(
        self,
        pretrained=True,
        weights=package_directory + "/weights/calc.caffemodel.pt",
    ):
        super().__init__()

        self.input_dim = (1, 120, 160)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=2, padding=4)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=1, padding=2)
        self.conv3 = nn.Conv2d(128, 4, kernel_size=(3, 3), stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.lrn1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)
        self.lrn2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)

        if pretrained:
            state_dict = torch.load(weights)
            my_new_state_dict = {}
            my_layers = list(self.state_dict().keys())
            for layer in my_layers:
                my_new_state_dict[layer] = state_dict[layer]
            self.load_state_dict(my_new_state_dict)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.lrn1(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.lrn2(x)

        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        return x


class ConvertToYUVandEqualizeHist:
    def __call__(self, img):
        img_yuv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return Image.fromarray(img_rgb)

############################ CALC VPR Model ####################################################
preprocess = transforms.Compose(
    [
        ConvertToYUVandEqualizeHist(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((120, 160), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ]
)

if not os.path.exists(package_directory + "/weights/calc.caffemodel.pt"):
    s3_bucket_download("placerecdata/weights/calc.caffemodel.pt", package_directory + "/weights/calc.caffemodel.pt")

class CALC(BaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        if pretrained:
            model = CalcModel(pretrained=True)
        else:
            model = CalcModel(pretrained=False)
        super().__init__(model=model, preprocess=preprocess, name="calc")

        if self.device == "mps":
            self.device = "cpu"

        self.model.to(self.device)
        self.model.eval()
