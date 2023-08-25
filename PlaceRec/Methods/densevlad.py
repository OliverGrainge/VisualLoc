import torch
from torchvision.models import inception_v3
from torchvision import transforms
from .base_method import BaseFunctionality
from ..utils import s3_bucket_download
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
from torchvision import models
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from ..utils import ImageDataset
from tqdm import tqdm

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="torchvision.models.inception"
)

package_directory = os.path.dirname(os.path.abspath(__file__))


class ModifiedInceptionV3(models.Inception3):
    def __init__(self):
        super().__init__()
        original_model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT
        )
        self.load_state_dict(original_model.state_dict())
        del original_model

    def forward(self, x):
        # Manually replicate the forward pass up to the Mixed_6e block
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        return x


class DenseVLADModel(nn.Module):
    def __init__(self, num_clusters: int = 248, feature_size: int = 64):
        super().__init__()
        self.feature_size = feature_size
        self.features = ModifiedInceptionV3()
        self.features.eval()
        self.codes = nn.Parameter(torch.randn(num_clusters, feature_size))
        self.num_clusters = self.codes.shape[0]
        self.visual_word_dim = self.codes.shape[1]

        try:
            self.load_state_dict(
                torch.load(package_directory + "/weights/densevlad_weights.pth")
            )
        except:
            print(
                "DenseVLAD Weights are not yet saved. Compute clusters before contiuing. They have only been randomly initialized"
            )
            pass

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def compute_clusters(
        self,
        img_paths: list = None,
        max_sample: int = 10000,
        batch_size: int = 32,
        save: bool = True,
        device: str = "cuda",
    ):
        dataset = ImageDataset(img_paths=img_paths, preprocess=self.preprocess)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        desc = []
        self.features.to(device)
        sample_size = int(int(max_sample / len(img_paths)) * batch_size)
        for batch in tqdm(loader):
            with torch.no_grad():
                feat = self.features(batch.to(device)).detach().cpu()
                feat = feat.reshape(-1, self.feature_size)
                # sample features
                sample_idx = torch.randint(0, feat.shape[0], size=(sample_size,))
                sample_feat = feat[sample_idx]
            desc.append(sample_feat.numpy())

        desc = np.vstack(desc)

        # cluster the descriptors
        print("=====> Clustering " + str(desc.shape[0]), " visual words")
        kmeans = KMeans(n_clusters=self.num_clusters, n_init="auto").fit(desc)
        clusters = torch.tensor(kmeans.cluster_centers_, requires_grad=False)
        self.codes = nn.Parameter(clusters)
        torch.save(
            self.state_dict(), package_directory + "/weights/densevlad_weights.pth"
        )
        self.to(device)

    def batch_vlad(self, features_batch):
        batch_size, n_descriptors, descriptor_dim = features_batch.shape

        # Compute distances for the entire batch
        # Shape: [batch_size, n_descriptors, n_clusters]
        distances = torch.cdist(
            features_batch, self.codes.unsqueeze(0).repeat(batch_size, 1, 1)
        )

        # Assignments for the entire batch
        # Shape: [batch_size, n_descriptors]
        assignments = torch.argmin(distances, dim=2)

        # Compute residuals for the entire batch
        # Shape: [batch_size, n_descriptors, descriptor_dim]
        residuals = features_batch - self.codes[assignments]

        # intranormalization before aggregation
        residuals = residuals / (torch.norm(residuals, dim=2, keepdim=True) + 1e-8)

        # Initialize batch vlad representation
        # Shape: [batch_size, n_clusters, descriptor_dim]
        batch_vlad_representation = torch.zeros(
            batch_size,
            self.codes.size(0),
            descriptor_dim,
            device=features_batch.device,
        )

        # Use scatter_add to aggregate residuals for the entire batch
        batch_vlad_representation.scatter_add_(
            1, assignments.unsqueeze(-1).expand_as(residuals), residuals
        )

        # Flatten and normalize
        batch_vlad_representation = batch_vlad_representation.view(batch_size, -1)

        # standardise
        mean = torch.mean(batch_vlad_representation, dim=1, keepdim=True)
        std = torch.std(batch_vlad_representation, dim=1, keepdim=True)
        batch_vlad_representation = (batch_vlad_representation - mean) / (std + 1e-8)

        return batch_vlad_representation

    def forward(self, x):
        with torch.no_grad():
            feat = self.features(x)
            # reshape into visual words
            visual_words = feat.reshape(feat.shape[0], -1, self.feature_size)

            # intra normalization

            visual_words = F.normalize(visual_words, dim=1)
            # compute vlad representation over batch

            vlad = self.batch_vlad(visual_words)
        return vlad


class DenseVLAD(BaseFunctionality):
    def __init__(self, num_clusters: int = 248, feature_size: int = 64):
        super().__init__()
        self.name = "densvlad"

        # amosnet layers not implemented on metal
        self.model = DenseVLADModel(
            feature_size=feature_size, num_clusters=num_clusters
        )
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def compute_clusters(
        self,
        img_paths: list = None,
        max_sample: int = 10000,
        batch_size: int = 32,
        save: bool = True,
        device: str = "cuda",
    ):
        self.model.compute_clusters(
            img_paths=img_paths,
            max_sample=max_sample,
            batch_size=batch_size,
            save=save,
            device=device,
        )

    def compute_query_desc(
        self,
        images: torch.Tensor = None,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
    ) -> dict:
        if images is not None and dataloader is None:
            with torch.no_grad():
                all_desc = self.model(images.to(self.device)).detach().cpu().numpy()
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(
                dataloader, desc="Computing DenseVLAD Query Desc", disable=not pbar
            ):
                with torch.no_grad():
                    all_desc.append(
                        self.model(batch.to(self.device)).detach().cpu().numpy()
                    )
            all_desc = np.vstack(all_desc)

        query_desc = {"query_descriptors": all_desc}
        self.set_query(query_desc)
        return query_desc

    def compute_map_desc(
        self,
        images: torch.Tensor = None,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
    ) -> dict:
        if images is not None and dataloader is None:
            with torch.no_grad():
                all_desc = self.model(images.to(self.device)).detach().cpu().numpy()
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(
                dataloader, desc="Computing DenseVLAD Map Desc", disable=not pbar
            ):
                with torch.no_grad():
                    all_desc.append(
                        self.model(batch.to(self.device)).detach().cpu().numpy()
                    )
            all_desc = np.vstack(all_desc)
        else:
            raise Exception("can only pass 'images' or 'dataloader'")

        map_desc = {"map_descriptors": all_desc}
        self.set_map(map_desc)
        return map_desc
