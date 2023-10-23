import os
import pickle
from typing import Tuple

import numpy as np
import torch
from torchvision import transforms
from torchvision.models import AlexNet_Weights
from tqdm import tqdm

from .base_method import BaseFunctionality

alexnet_directory = os.path.dirname(os.path.abspath(__file__))


class AlexNet(BaseFunctionality):
    def __init__(self):
        # Name of technique
        self.name = "alexnet"

        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "alexnet",
            weights=AlexNet_Weights.DEFAULT,
            verbose=False,
        )
        self.model = self.model.features[:7]

        # send the model to relevant accelerator
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Dimensions to project into
        self.nDims = 4096

        # elimminate gradient computations and send to accelerator
        self.model.to(self.device)
        self.model.eval()

        # preprocess for network
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([224, 244], antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.map = None
        self.map_desc = None
        self.query_desc = None

    def compute_query_desc(
        self,
        images: torch.Tensor = None,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
    ) -> dict:
        if images is not None and dataloader is None:
            desc = self.model(images.to(self.device)).detach().cpu().numpy()
            Ds = desc.reshape([images.shape[0], -1])  # flatten
            rng = np.random.default_rng(seed=0)
            Proj = rng.standard_normal([Ds.shape[1], self.nDims], "float32")
            Proj = Proj / np.linalg.norm(Proj, axis=1, keepdims=True)
            Ds = Ds @ Proj
            all_desc = Ds
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(dataloader, desc="Computing AlexNet Query Desc", disable=not pbar):
                desc = self.model(batch.to(self.device)).detach().cpu().numpy()
                Ds = desc.reshape([batch.shape[0], -1])  # flatten
                rng = np.random.default_rng(seed=0)
                Proj = rng.standard_normal([Ds.shape[1], self.nDims], "float32")
                Proj = Proj / np.linalg.norm(Proj, axis=1, keepdims=True)
                Ds = Ds @ Proj
                all_desc.append(Ds)
            all_desc = np.vstack(all_desc)
        else:
            raise Exception("Can only pass 'images' or 'dataloader'")

        query_desc = {"query_descriptors": all_desc}
        self.set_query(query_desc)
        return query_desc

    def compute_map_desc(
        self,
        images: np.ndarray = None,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
    ) -> dict:
        if images is not None and dataloader is None:
            desc = self.model(images.to(self.device)).detach().cpu().numpy()
            Ds = desc.reshape([images.shape[0], -1])  # flatten
            rng = np.random.default_rng(seed=0)
            Proj = rng.standard_normal([Ds.shape[1], self.nDims], "float32")
            Proj = Proj / np.linalg.norm(Proj, axis=1, keepdims=True)
            Ds = Ds @ Proj
            all_desc = Ds
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(dataloader, desc="Computing AlexNet Map Desc", disable=not pbar):
                desc = self.model(batch.to(self.device)).detach().cpu().numpy()
                Ds = desc.reshape([batch.shape[0], -1])  # flatten
                rng = np.random.default_rng(seed=0)
                Proj = rng.standard_normal([Ds.shape[1], self.nDims], "float32")
                Proj = Proj / np.linalg.norm(Proj, axis=1, keepdims=True)
                Ds = Ds @ Proj
                all_desc.append(Ds)
            all_desc = np.vstack(all_desc)
        else:
            raise Exception("Can only pass 'images' or 'dataloader'")

        map_desc = {"map_descriptors": all_desc}
        self.set_map(map_desc)
        return map_desc
