import os
import pickle
import sys
from typing import Tuple

import numpy as np
import sklearn
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from tqdm import tqdm
from torch import nn

from .base_method import BaseModelWrapper

cosplace_directory = os.path.dirname(os.path.abspath(__file__))


original_stdout = sys.stdout
sys.stdout = open(os.devnull, "w")
model = torch.hub.load(
    "gmberton/cosplace",
    "get_trained_model",
    backbone="ResNet50",
    fc_output_dim=2048,
    verbose=False,
)
sys.stdout = original_stdout

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((512, 512), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Example: Kaiming Initialization for Conv2D and Linear layers
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class CosPlace(BaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        if not pretrained:
            model.apply(init_weights)
        super().__init__(model=model, preprocess=preprocess, name="cosplace")
