import math
import os
from os.path import join

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from sklearn.cluster import KMeans
from tqdm import tqdm

from PlaceRec.Datasets import Pitts30k
from PlaceRec.Methods import SingleStageBaseModelWrapper
from PlaceRec.utils import L2Norm, get_config

config = get_config()


class ResNet(nn.Module):
    def __init__(
        self,
        model_name="resnet50",
        pretrained=True,
        layers_to_freeze=2,
        layers_to_crop=[],
        output_dim=128,
    ):
        """Class representing the resnet backbone used in the pipeline
        we consider resnet network as a list of 5 blocks (from 0 to 4),
        layer 0 is the first conv+bn and the other layers (1 to 4) are the rest of the residual blocks
        we don't take into account the global pooling and the last fc

        Args:
            model_name (str, optional): The architecture of the resnet backbone to instanciate. Defaults to 'resnet50'.
            pretrained (bool, optional): Whether pretrained or not. Defaults to True.
            layers_to_freeze (int, optional): The number of residual blocks to freeze (starting from 0) . Defaults to 2.
            layers_to_crop (list, optional): Which residual layers to crop, for example [3,4] will crop the third and fourth res blocks. Defaults to [].

        Raises:
            NotImplementedError: if the model_name corresponds to an unknown architecture.
        """
        super().__init__()
        self.model_name = model_name.lower()
        self.layers_to_freeze = layers_to_freeze

        if pretrained:
            # the new naming of pretrained weights, you can change to V2 if desired.
            weights = "IMAGENET1K_V1"
        else:
            weights = None

        if "swsl" in model_name or "ssl" in model_name:
            # These are the semi supervised and weakly semi supervised weights from Facebook
            self.model = torch.hub.load(
                "facebookresearch/semi-supervised-ImageNet1K-models", model_name
            )
        else:
            if "resnext50" in model_name:
                self.model = torchvision.models.resnext50_32x4d(weights=weights)
            elif "resnet50" in model_name:
                self.model = torchvision.models.resnet50(weights=weights)
            elif "101" in model_name:
                self.model = torchvision.models.resnet101(weights=weights)
            elif "152" in model_name:
                self.model = torchvision.models.resnet152(weights=weights)
            elif "34" in model_name:
                self.model = torchvision.models.resnet34(weights=weights)
            elif "18" in model_name:
                # self.model = torchvision.models.resnet18(pretrained=False)
                self.model = torchvision.models.resnet18(weights=weights)
            elif "wide_resnet50_2" in model_name:
                self.model = torchvision.models.wide_resnet50_2(weights=weights)
            else:
                raise NotImplementedError("Backbone architecture not recognized!")

        # freeze only if the model is pretrained
        if pretrained:
            if layers_to_freeze >= 0:
                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)
            if layers_to_freeze >= 1:
                self.model.layer1.requires_grad_(False)
            if layers_to_freeze >= 2:
                self.model.layer2.requires_grad_(False)
            if layers_to_freeze >= 3:
                self.model.layer3.requires_grad_(False)

        # remove the avgpool and most importantly the fc layer
        self.model.avgpool = None
        self.model.fc = None

        if 4 in layers_to_crop:
            self.model.layer4 = None
        if 3 in layers_to_crop:
            self.model.layer3 = None

        out_channels = 2048
        if "34" in model_name or "18" in model_name:
            out_channels = 512

        self.out_channels = (
            out_channels // 2 if self.model.layer4 is None else out_channels
        )
        self.out_channels = (
            self.out_channels // 2 if self.model.layer3 is None else self.out_channels
        )

        self.channel_pool = nn.Conv2d(
            self.out_channels, 128, kernel_size=(1, 1), bias=False
        )

        self.out_channels = 128

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        if self.model.layer3 is not None:
            x = self.model.layer3(x)
        if self.model.layer4 is not None:
            x = self.model.layer4(x)
        x = self.channel_pool(x)
        return x


class NetVLADagg(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(
        self,
        clusters_num=64,
        dim=128,
        in_channels=1024,
        normalize_input=True,
        work_with_tokens=False,
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
        self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))

    def init_params(self, centroids, descriptors):
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

    def forward(self, x):
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

    def initialize_netvlad_layer(self, cluster_ds, backbone):
        descriptors_num = 50000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        random_sampler = SubsetRandomSampler(
            np.random.choice(len(cluster_ds), images_num, replace=False)
        )
        random_dl = DataLoader(
            dataset=cluster_ds,
            num_workers=config["train"]["num_workers"],
            batch_size=config["train"]["batch_size"],
            sampler=random_sampler,
        )
        with torch.no_grad():
            descriptors = np.zeros(shape=(descriptors_num, self.dim), dtype=np.float32)
            for iteration, (idx, inputs) in enumerate(tqdm(random_dl, ncols=100)):
                inputs = inputs.to(next(backbone.parameters()).device)
                outputs = backbone(inputs)
                norm_outputs = F.normalize(outputs, p=2, dim=1)
                image_descriptors = norm_outputs.view(
                    norm_outputs.shape[0], self.dim, -1
                ).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = (
                    iteration * config["train"]["batch_size"] * descs_num_per_image
                )
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(
                        image_descriptors.shape[1], descs_num_per_image, replace=False
                    )
                    startix = batchix + ix * descs_num_per_image
                    descriptors[
                        startix : startix + descs_num_per_image, :
                    ] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(self.dim, self.clusters_num, niter=100, verbose=False)
        kmeans.train(descriptors)
        self.init_params(kmeans.centroids, descriptors)
        self = self.to((next(backbone.parameters()).device))


class NetVLADNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer."""

    def __init__(self):
        super().__init__()
        self.backbone = ResNet(
            model_name="resnet50",
            pretrained=True,
            layers_to_freeze=1,
            layers_to_crop=[4],
        )
        self.aggregation = NetVLADagg(clusters_num=64, dim=128, work_with_tokens=False)
        self.norm = L2Norm()

    def forward(self, x, norm: bool = True):
        x = self.backbone(x)
        x = self.aggregation(x)
        if norm:
            x = self.norm(x)
        return x


class ResNet34_NetVLADNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer."""

    def __init__(self):
        super().__init__()
        self.backbone = ResNet(
            model_name="resnet34",
            pretrained=True,
            layers_to_freeze=1,
            layers_to_crop=[4],
        )
        self.aggregation = NetVLADagg(
            clusters_num=64, dim=128, in_channels=256, work_with_tokens=False
        )

        self.norm = L2Norm()

    def forward(self, x, norm: bool = True):
        x = self.backbone(x)
        x = self.aggregation(x)
        if norm:
            x = self.norm(x)
        return x


class ResNet18_NetVLADNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer."""

    def __init__(self):
        super().__init__()
        self.backbone = ResNet(
            model_name="resnet18",
            pretrained=True,
            layers_to_freeze=1,
            layers_to_crop=[4],
        )
        self.aggregation = NetVLADagg(
            clusters_num=64, dim=128, in_channels=256, work_with_tokens=False
        )

        self.norm = L2Norm()

    def forward(self, x, norm: bool = True):
        x = self.backbone(x)
        x = self.aggregation(x)
        if norm:
            x = self.norm(x)
        return x


class MobileNetV2_NetVLADNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer."""

    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.mobilenet_v2(pretrained=True).features[:-1]
        self.aggregation = NetVLADagg(
            clusters_num=64, dim=128, in_channels=320, work_with_tokens=False
        )
        self.norm = L2Norm()

    def forward(self, x, norm: bool = True):
        x = self.backbone(x)
        x = self.aggregation(x)
        if norm:
            x = self.norm(x)
        return x


preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((320, 320), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class NetVLAD(SingleStageBaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        self.model = NetVLADNet()
        if not pretrained:
            if torch.cuda.is_available():
                self.model.to("cuda")
            elif torch.backends.mps.is_available():
                self.model.to("mps")
            else:
                self.model.to("cpu")
            ds = Pitts30k()
            dl = ds.query_images_loader(preprocess=preprocess)
            cluster_ds = dl.dataset
            self.model.aggregation.initialize_netvlad_layer(
                cluster_ds, self.model.backbone
            )
            self.model.to("cpu")

        name = "netvlad"
        weight_path = join(config["weights_directory"], name + ".ckpt")
        if pretrained:
            if not os.path.exists(weight_path):
                raise Exception(f"Could not find weights at {weight_path}")
            self.load_weights(weight_path)
            super().__init__(
                model=self.model,
                preprocess=preprocess,
                name=name,
                weight_path=weight_path,
            )
        else:
            super().__init__(
                model=self.model, preprocess=preprocess, name=name, weight_path=None
            )


class MobileNetV2_NetVLAD(SingleStageBaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        self.model = MobileNetV2_NetVLADNet()
        if not pretrained:
            if torch.cuda.is_available():
                self.model.to("cuda")
            elif torch.backends.mps.is_available():
                self.model.to("mps")
            else:
                self.model.to("cpu")
            ds = Pitts30k()
            dl = ds.query_images_loader(preprocess=preprocess)
            cluster_ds = dl.dataset
            self.model.aggregation.initialize_netvlad_layer(
                cluster_ds, self.model.backbone
            )
            self.model.to("cpu")

        name = "mobilenetv2_netvlad"
        weight_path = join(config["weights_directory"], name + ".ckpt")
        if pretrained:
            if not os.path.exists(weight_path):
                raise Exception(f"Could not find weights at {weight_path}")
            self.load_weights(weight_path)
            super().__init__(
                model=self.model,
                preprocess=preprocess,
                name=name,
                weight_path=weight_path,
            )
        else:
            super().__init__(
                model=self.model, preprocess=preprocess, name=name, weight_path=None
            )


class ResNet34_NetVLAD(SingleStageBaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        self.model = ResNet34_NetVLADNet()
        if not pretrained:
            if torch.cuda.is_available():
                self.model.to("cuda")
            elif torch.backends.mps.is_available():
                self.model.to("mps")
            else:
                self.model.to("cpu")
            ds = Pitts30k()
            dl = ds.query_images_loader(preprocess=preprocess)
            cluster_ds = dl.dataset
            self.model.aggregation.initialize_netvlad_layer(
                cluster_ds, self.model.backbone
            )
            self.model.to("cpu")

        name = "resnet34_netvlad"
        weight_path = join(config["weights_directory"], name + ".ckpt")
        if pretrained:
            if not os.path.exists(weight_path):
                raise Exception(f"Could not find weights at {weight_path}")
            self.load_weights(weight_path)
            super().__init__(
                model=self.model,
                preprocess=preprocess,
                name=name,
                weight_path=weight_path,
            )
        else:
            super().__init__(
                model=self.model, preprocess=preprocess, name=name, weight_path=None
            )


class ResNet18_NetVLAD(SingleStageBaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        self.model = NetVLADNet()
        if not pretrained:
            if torch.cuda.is_available():
                self.model.to("cuda")
            elif torch.backends.mps.is_available():
                self.model.to("mps")
            else:
                self.model.to("cpu")
            ds = Pitts30k()
            dl = ds.query_images_loader(preprocess=preprocess)
            cluster_ds = dl.dataset
            self.model.aggregation.initialize_netvlad_layer(
                cluster_ds, self.model.backbone
            )
            self.model.to("cpu")

        name = "resnet18_netvlad"
        weight_path = join(config["weights_directory"], name + ".ckpt")
        if pretrained:
            if not os.path.exists(weight_path):
                raise Exception(f"Could not find weights at {weight_path}")
            self.load_weights(weight_path)
            super().__init__(
                model=self.model,
                preprocess=preprocess,
                name=name,
                weight_path=weight_path,
            )
        else:
            super().__init__(
                model=self.model, preprocess=preprocess, name=name, weight_path=None
            )
