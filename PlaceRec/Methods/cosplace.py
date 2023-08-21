from .base_method import BaseFunctionality
import torch
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
import numpy as np
from typing import Tuple
import os
from tqdm import tqdm
import pickle

cosplace_directory = os.path.dirname(os.path.abspath(__file__))


class CosPlace(BaseFunctionality):
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=2048,
            verbose=False,
        ).to(self.device)
        self.model.eval()

        # preprocess images for method (None required)
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((512, 512), antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # initialize the map to None
        self.map = None
        self.map_desc = None
        self.query_desc = None
        # method name
        self.name = "cosplace"

    def compute_query_desc(
        self, images: torch.Tensor = None, dataloader=None, pbar: bool = True
    ) -> dict:
        if images is not None and dataloader is None:
            all_desc = self.model(images.to(self.device)).detach().cpu().numpy()
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(
                dataloader, desc="Computing CosPlace Query Desc", disable=not pbar
            ):
                all_desc.append(
                    self.model(batch.to(self.device)).detach().cpu().numpy()
                )
            all_desc = np.vstack(all_desc)

        query_desc = {"query_descriptors": all_desc}
        self.set_query(query_desc)
        return query_desc

    def compute_map_desc(
        self, images: torch.Tensor = None, dataloader=None, pbar: bool = True
    ) -> dict:
        if images is not None and dataloader is None:
            all_desc = self.model(images.to(self.device)).detach().cpu().numpy()
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(
                dataloader, desc="Computing CosPlace Map Desc", disable=not pbar
            ):
                all_desc.append(
                    self.model(batch.to(self.device)).detach().cpu().numpy()
                )
            all_desc = np.vstack(all_desc)
        else:
            raise Exception("can only pass 'images' or 'dataloader'")

        map_desc = {"map_descriptors": all_desc}
        self.set_map(map_desc)
        return map_desc
