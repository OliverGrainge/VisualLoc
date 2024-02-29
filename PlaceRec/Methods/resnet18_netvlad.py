import os
import pickle
import zipfile
from argparse import Namespace
from os.path import join
from typing import Tuple

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm

from ..utils import get_config
from .base_method import BaseModelWrapper

netvlad_directory = os.path.dirname(os.path.abspath(__file__))
config = get_config()


class NetVLAD_Aggregation(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(
        self,
        clusters_num: int = 64,
        dim: int = 128,
        normalize_input: bool = True,
        work_with_tokens: bool = False,
    ):
        """
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.clusters_num = clusters_num
        self.dim = dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.work_with_tokens = work_with_tokens
        if work_with_tokens:
            self.conv = nn.Conv1d(dim, clusters_num, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))

    def init_params(self, centroids: np.ndarray, descriptors: np.ndarray) -> None:
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        if self.work_with_tokens:
            self.conv.weight = nn.Parameter(
                torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2)
            )
        else:
            self.conv.weight = nn.Parameter(
                torch.from_numpy(self.alpha * centroids_assign)
                .unsqueeze(2)
                .unsqueeze(3)
            )
        self.conv.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.work_with_tokens:
            x = x.permute(0, 2, 1)
            N, D, _ = x.shape[:]
        else:
            N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = torch.zeros(
            [N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device
        )
        for D in range(
            self.clusters_num
        ):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - self.centroids[
                D : D + 1, :
            ].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:, D : D + 1, :].unsqueeze(2)
            vlad[:, D : D + 1, :] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

    def initialize_netvlad_layer(
        self, args: Namespace, cluster_ds: Dataset, backbone: nn.Module
    ) -> None:
        descriptors_num = 50000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        random_sampler = SubsetRandomSampler(
            np.random.choice(len(cluster_ds), images_num, replace=False)
        )
        random_dl = DataLoader(
            dataset=cluster_ds,
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size,
            sampler=random_sampler,
        )
        with torch.no_grad():
            backbone = backbone.eval()
            descriptors = np.zeros(
                shape=(descriptors_num, args.features_dim), dtype=np.float32
            )
            for iteration, (inputs, _) in enumerate(tqdm(random_dl, ncols=100)):
                inputs = inputs.to(args.device)
                outputs = backbone(inputs)
                norm_outputs = F.normalize(outputs, p=2, dim=1)
                image_descriptors = norm_outputs.view(
                    norm_outputs.shape[0], args.features_dim, -1
                ).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * args.infer_batch_size * descs_num_per_image
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(
                        image_descriptors.shape[1], descs_num_per_image, replace=False
                    )
                    startix = batchix + ix * descs_num_per_image
                    descriptors[
                        startix : startix + descs_num_per_image, :
                    ] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(
            args.features_dim, self.clusters_num, niter=100, verbose=False
        )
        kmeans.train(descriptors)
        self.init_params(kmeans.centroids, descriptors)
        self = self.to(args.device)


class ResNet_NetVLAD(nn.Module):
    def __init__(self):
        super().__init__()
        # get the backbone
        backbone = resnet18(weights=None)
        layers = list(backbone.children())[:-3]
        self.backbone = nn.Sequential(*layers)
        # get the aggregation
        self.aggregation = NetVLAD_Aggregation(dim=256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


# ============================================================= NetVLAD ==================================================================
preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((480, 640), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ResNet18_NetVLAD(BaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        model = ResNet_NetVLAD()
        if pretrained:
            weights_pth = join(
                config["weights_directory"], "msls_r18l3_netvlad_partial.pth"
            )
            if not os.path.exists(weights_pth):
                raise Exception(f"Could not find weights at {weights_pth}")
            model.load_state_dict(torch.load(weights_pth))

        super().__init__(model=model, preprocess=preprocess, name="resnet18_netvlad")
