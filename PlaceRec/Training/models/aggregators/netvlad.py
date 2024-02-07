import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from torch.utils.data import DataLoader, TensorDataset
import faiss
import numpy as np
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from tqdm import tqdm
from .l2norm import L2Norm


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(
        self, feature_map_shape, out_dim=1024, clusters_num=64, dim=128, normalize_input=True, work_with_tokens=False
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
        self.channel_pool = nn.Conv2d(feature_map_shape[0], dim, kernel_size=(1,1))
        self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1))
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))
        self.projection = nn.Linear(int(dim * clusters_num), out_dim)
        self.norm = L2Norm()

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        self.conv.weight = nn.Parameter(
            torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2).unsqueeze(3)
        )

    def initialize_netvlad_layer(self, cluster_ds, backbone):
        descriptors_num = 50000
        descs_num_per_image = 40
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        random_sampler = SubsetRandomSampler(
            np.random.choice(len(cluster_ds), images_num, replace=False)
        )
        random_dl = DataLoader(
            dataset=cluster_ds,
            num_workers=4,
            batch_size=10,
            sampler=random_sampler,
        )
        with torch.no_grad():
            backbone = backbone.eval().cuda()
            self.channel_pool.eval().cuda()
            descriptors = np.zeros(
                shape=(descriptors_num, self.dim), dtype=np.float32
            )
            for iteration, (inputs, _) in enumerate(tqdm(random_dl, ncols=100)):
                inputs = inputs.to("cuda")
                outputs = self.channel_pool(backbone(inputs))
                norm_outputs = F.normalize(outputs, p=2, dim=1)
                image_descriptors = norm_outputs.view(
                    norm_outputs.shape[0], self.dim, -1
                ).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * 10 * descs_num_per_image
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(
                        image_descriptors.shape[1], descs_num_per_image, replace=False
                    )
                    startix = batchix + ix * descs_num_per_image
                    descriptors[
                        startix : startix + descs_num_per_image, :
                    ] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(
            self.dim, self.clusters_num, niter=100, verbose=False
        )
        kmeans.train(descriptors)
        self.init_params(kmeans.centroids, descriptors)
        self = self.to("cuda")

    def set_output_dim(self, args):
        if args.fc_output_dim is not None:
            args.features_dim = args.fc_output_dim

    def forward(self, x):
        x = self.channel_pool(x)
        N, D, H, W = x.shape[:]
        x = self.norm(x)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        # Compute residuals for all clusters simultaneously
        residual = x_flatten.unsqueeze(1) - self.centroids.unsqueeze(0).unsqueeze(-1)
        residual *= soft_assign.unsqueeze(2)

        vlad = residual.sum(dim=-1)
        vlad = self.norm(vlad)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = self.projection(vlad)
        vlad = self.norm(vlad)  # L2 normalize
        return vlad



class NetVLADTokens(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(
        self, feature_map_shape, out_dim=1024, clusters_num=64, dim=128, normalize_input=True, work_with_tokens=False
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
        self.token_pool = nn.Linear(feature_map_shape[1], dim)
        self.soft_lin = nn.Linear(dim, clusters_num)
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))
        self.projection = nn.Linear(int(dim * clusters_num), out_dim)
        self.norm = L2Norm()

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        self.conv.weight = nn.Parameter(
            torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2).unsqueeze(3)
        )

    def initialize_netvlad_layer(self, cluster_ds, backbone):
        descriptors_num = 50000
        descs_num_per_image = 40
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        random_sampler = SubsetRandomSampler(
            np.random.choice(len(cluster_ds), images_num, replace=False)
        )
        random_dl = DataLoader(
            dataset=cluster_ds,
            num_workers=4,
            batch_size=10,
            sampler=random_sampler,
        )
        with torch.no_grad():
            backbone = backbone.eval().cuda()
            self.channel_pool.eval().cuda()
            descriptors = np.zeros(
                shape=(descriptors_num, self.dim), dtype=np.float32
            )
            for iteration, (inputs, _) in enumerate(tqdm(random_dl, ncols=100)):
                inputs = inputs.to("cuda")
                outputs = self.token_pool(backbone(inputs))
                outputs = outputs.permute(0, 2, 1)
                norm_outputs = F.normalize(outputs, p=2, dim=1)
                image_descriptors = norm_outputs.view(
                    norm_outputs.shape[0], self.dim, -1
                ).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * 10 * descs_num_per_image
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(
                        image_descriptors.shape[1], descs_num_per_image, replace=False
                    )
                    startix = batchix + ix * descs_num_per_image
                    descriptors[
                        startix : startix + descs_num_per_image, :
                    ] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(
            self.dim, self.clusters_num, niter=100, verbose=False
        )
        kmeans.train(descriptors)
        self.init_params(kmeans.centroids, descriptors)
        self = self.to("cuda")

    def set_output_dim(self, args):
        if args.fc_output_dim is not None:
            args.features_dim = args.fc_output_dim

    def forward(self, x):
        x = self.token_pool(x)
        x = x.permute(0, 2, 1)
        B, N, D = x.shape
        x = self.norm(x)  # Across descriptor dim
        x = x.permute(0, 2, 1)
        soft_assign = self.soft_lin(x)
        soft_assign = soft_assign.permute(0, 2, 1)
        soft_assign = F.softmax(soft_assign, dim=1)
        x = x.permute(0, 2, 1)
        # Compute residuals for all clusters simultaneously
        residual = x.unsqueeze(1) - self.centroids.unsqueeze(0).unsqueeze(-1)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)
        vlad = self.norm(vlad)  # intra-normalization
        vlad = vlad.view(B, -1)  # Flatten
        vlad = self.projection(vlad)
        vlad = self.norm(vlad)  # L2 normalize
        return vlad


