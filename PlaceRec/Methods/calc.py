import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import cv2
import random
import numpy as np
import torch.nn.functional as F
import os
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from glob import glob
from PIL import Image
from .base_method import BaseFunctionality
from typing import Tuple
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sklearn
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from ..utils import s3_bucket_download


package_directory = os.path.dirname(os.path.abspath(__file__))


class CalcModel(nn.Module):
    def __init__(
        self,
        pretrained=True,
        weights="/home/oliver/Documents/github/VisualLoc/PlaceRec/Methods/weights/calc.caffemodel.pt",
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


class CALC(BaseFunctionality):
    def __init__(self):
        super().__init__()  

        if not os.path.exists(package_directory + '/weights/calc.caffemodel.pt'):
            s3_bucket_download("placerecdata/weights/calc.caffemodel.pt", package_directory + "/weights/calc.caffemodel.pt")

        self.model = CalcModel(pretrained=True).to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                ConvertToYUVandEqualizeHist(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((120, 160), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
            ]
        )

        self.map = None
        self.map_desc = None
        self.query_desc = None
        self.name = "calc"

    def compute_query_desc(
        self,
        images: torch.Tensor = None,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
    ) -> dict:
        if images is not None and dataloader is None:
            all_desc = self.model(images.to(self.device)).detach().cpu().numpy()
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(
                dataloader, desc="Computing CALC Query Desc", disable=not pbar
            ):
                all_desc.append(
                    self.model(batch.to(self.device)).detach().cpu().numpy()
                )
            all_desc = np.vstack(all_desc)

        query_desc = {"query_descriptors": all_desc / np.linalg.norm(all_desc, axis=0)}
        self.set_query(query_desc)
        return query_desc

    def compute_map_desc(
        self,
        images: torch.Tensor = None,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
    ) -> dict:
        if images is not None and dataloader is None:
            all_desc = self.model(images.to(self.device)).detach().cpu().numpy()
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(
                dataloader, desc="Computing CALC Map Desc", disable=not pbar
            ):
                all_desc.append(
                    self.model(batch.to(self.device)).detach().cpu().numpy()
                )
            all_desc = np.vstack(all_desc)
        else:
            raise Exception("can only pass 'images' or 'dataloader'")

        map_desc = {"map_descriptors": all_desc / np.linalg.norm(all_desc, axis=0)}
        self.set_map(map_desc)

        return map_desc
