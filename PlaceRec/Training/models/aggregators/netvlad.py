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
        x = self.token_pool(x)
        B, N, D = x.shape()
        x = self.norm(x)  # Across descriptor dim
        soft_assign = self.conv(x).view(B, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        # Compute residuals for all clusters simultaneously
        residual = x.unsqueeze(1) - self.centroids.unsqueeze(0).unsqueeze(-1)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)
        vlad = self.norm(vlad)  # intra-normalization
        vlad = vlad.view(B, -1)  # Flatten
        vlad = self.projection(vlad)
        vlad = self.norm(vlad)  # L2 normalize
        return vlad

"""
class NetVLAD(nn.Module):

    def __init__(self, feature_map_shape: torch.tensor, out_dim: int=1024, clusters_num=64, dim=128):#, #work_with_tokens=False):
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        super().__init__()
        self.clusters_num = clusters_num
        self.dim = dim
        self.alpha = 0
        #self.work_with_tokens = work_with_tokens
        self.channel_pool = nn.Conv2d(in_channels=feature_map_shape[0], out_channels=dim, kernel_size=1, bias=True)
        #if work_with_tokens:
        #    self.conv = nn.Conv1d(dim, clusters_num, kernel_size=1, bias=False)
        #else:
        self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1))
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))
        self.projection = nn.Linear(dim * clusters_num, out_dim)

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        #if self.work_with_tokens:
        #    self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2))
        #else:
        self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*centroids_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None
        self.norm = L2Norm()


    def forward(self, x):
        x = self.channel_pool(x)
        N, D, H, W = x.shape[:]
        x = self.norm(x)  # Across descriptor dim
        #print(x.shape)
        x_flatten = x.view(N, D, H * W)
        #soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        #soft_assign = F.softmax(soft_assign, dim=1)
        
        # Compute residuals for all clusters simultaneously
        #residual = x_flatten.unsqueeze(1) - self.centroids.unsqueeze(0).unsqueeze(-1)
        #residual *= soft_assign.unsqueeze(2)
        
        #vlad = residual.sum(dim=-1)
        #vlad = F.normalize(vlad, p=2.0, dim=2)  # intra-normalization
        #vlad = vlad.view(N, -1)  # Flatten
        #vlad = F.normalize(vlad, p=2.0, dim=1)  # L2 normalize
        
        return x_flatten

    def forward(self, x):
        x = self.channel_pool(x)

        #if self.work_with_tokens:
        #    x = x.permute(0, 2, 1)
        #    N, D, _ = x.shape[:]
        #else:
        N, D, H, W = x.shape[:]
        x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        #x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x)
        #soft_assign = soft_assign.view(N, 64, -1)
        #soft_assign = F.softmax(soft_assign, dim=1)
        #vlad = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)
        #for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
        #    residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
        #            self.centroids[D:D+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        #    residual = residual * soft_assign[:,D:D+1,:].unsqueeze(2)
        #    vlad[:,D:D+1,:] = residual.sum(dim=-1)
        #vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        #vlad = vlad.view(N, -1)  # Flatten
        #vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        #vlad = self.projection(vlad)
        #vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return soft_assign


    def initialize_netvlad_layer(self, feature_maps: torch.tensor):
        ds = TensorDataset(feature_maps)
        dl = DataLoader(ds, batch_size=20)
        features = []
        with torch.no_grad():
            for batch in dl:
                features.append(self.channel_pool(batch[0]).detach().cpu())
        feature_maps = torch.vstack(features)
        descriptors = feature_maps.permute(0, 2, 3, 1).reshape(-1, self.dim).float().numpy()
        kmeans = faiss.Kmeans(descriptors.shape[1], self.clusters_num, niter=100, verbose=False)
        kmeans.train(descriptors)
        self.init_params(kmeans.centroids, descriptors)
"""