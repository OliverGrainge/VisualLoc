from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from .base_method import BaseTechnique
from typing import Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import sklearn
import zipfile
from ..utils import dropbox_download_file
from ..db_key import DROPBOX_ACCESS_TOKEN
import pickle
from tqdm import tqdm


netvlad_directory = os.path.dirname(os.path.abspath(__file__))


class NetVLAD_Aggregation(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, clusters_num=64, dim=128, normalize_input=True, work_with_tokens=False):
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

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        if self.work_with_tokens:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2))
        else:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*centroids_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
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
        vlad = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[D:D+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:,D:D+1,:].unsqueeze(2)
            vlad[:,D:D+1,:] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

    def initialize_netvlad_layer(self, args, cluster_ds, backbone):
        descriptors_num = 50000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        random_sampler = SubsetRandomSampler(np.random.choice(len(cluster_ds), images_num, replace=False))
        random_dl = DataLoader(dataset=cluster_ds, num_workers=args.num_workers,
                                batch_size=args.infer_batch_size, sampler=random_sampler)
        with torch.no_grad():
            backbone = backbone.eval()
            logging.debug("Extracting features to initialize NetVLAD layer")
            descriptors = np.zeros(shape=(descriptors_num, args.features_dim), dtype=np.float32)
            for iteration, (inputs, _) in enumerate(tqdm(random_dl, ncols=100)):
                inputs = inputs.to(args.device)
                outputs = backbone(inputs)
                norm_outputs = F.normalize(outputs, p=2, dim=1)
                image_descriptors = norm_outputs.view(norm_outputs.shape[0], args.features_dim, -1).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * args.infer_batch_size * descs_num_per_image
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(image_descriptors.shape[1], descs_num_per_image, replace=False)
                    startix = batchix + ix * descs_num_per_image
                    descriptors[startix:startix + descs_num_per_image, :] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(args.features_dim, self.clusters_num, niter=100, verbose=False)
        kmeans.train(descriptors)
        logging.debug(f"NetVLAD centroids shape: {kmeans.centroids.shape}")
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
        

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x



# ============================================================= NetVLAD ==================================================================



class NetVLAD(BaseTechnique):
    def __init__(self):

        self.name = "netvlad"
        self.model = ResNet_NetVLAD()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # load the model weights
        try:
            self.model.load_state_dict(torch.load(netvlad_directory + "/weights/msls_r18l3_netvlad_partial.pth"))
        except:
            print("===> downloading netvlad weights")
            dropbox_download_file("/weights/msls_r18l3_netvlad_partial.pth", netvlad_directory + "/weights/msls_r18l3_netvlad_partial.pth")
            self.model.load_state_dict(torch.load(netvlad_directory + "/weights/msls_r18l3_netvlad_partial.pth"))

        # send model to accelerator
        self.model.to(self.device)
        self.model.eval()

        # define image preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])



    def compute_query_desc(self, images: torch.Tensor = None, dataloader: torch.utils.data.dataloader.DataLoader = None, pbar: bool=True) -> dict:
        if images is not None and dataloader is None:
            all_desc = self.model(images.to(self.device)).detach().cpu().numpy()
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(dataloader, desc="Computing NetVLAD Query Desc", disable=not pbar):
                all_desc.append(self.model(batch.to(self.device)).detach().cpu().numpy())
            all_desc = np.vstack(all_desc)
        else: 
            raise Exception("Can only pass 'images' or 'dataloader'")

        query_desc = {"query_descriptors": all_desc}
        self.set_query(query_desc)
        return query_desc


    def compute_map_desc(self, images: torch.Tensor = None, dataloader: torch.utils.data.dataloader.DataLoader = None, pbar: bool=True) -> dict:
        if images is not None and dataloader is None:
            all_desc = self.model(images.to(self.device)).detach().cpu().numpy()
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(dataloader, desc="Computing NetVLAD Map Desc", disable=not pbar):
                all_desc.append(self.model(batch.to(self.device)).detach().cpu().numpy())
            all_desc = np.vstack(all_desc)
        else: 
            raise Exception("Can only pass 'images' or 'dataloader'")

        map_desc = {"map_descriptors": all_desc}
        self.set_map(map_desc)
        return map_desc

    def set_map(self, map_descriptors: dict) -> None:
        self.map_desc = map_descriptors
        try: 
            self.map = faiss.IndexFlatIP(map_descriptors["map_descriptors"].shape[1])
            faiss.normalize_L2(map_descriptors["map_descriptors"])
            self.map.add(map_descriptors["map_descriptors"])

        except: 
            self.map = NearestNeighbors(n_neighbors=10, algorithm='brute', 
                                        metric='cosine').fit(map_descriptors["map_descriptors"])

    def set_query(self, query_descriptors: dict) -> None:
        self.query_desc = query_descriptors


    def place_recognise(self, images: torch.Tensor=None, dataloader: torch.utils.data.dataloader.DataLoader = None, top_n: int=1, pbar: bool=True) -> Tuple[np.ndarray, np.ndarray]:
        desc = self.compute_query_desc(images=images, dataloader=dataloader, pbar=pbar)
        if isinstance(self.map, sklearn.neighbors._unsupervised.NearestNeighbors):
            dist, idx = self.map.kneighbors(desc["query_descriptors"])
            print("===================", dist)
            return idx[:, :top_n], 1 - dist[:, :top_n]
        else: 
            faiss.normalize_L2(desc["query_descriptors"])
            dist, idx = self.map.search(desc["query_descriptors"], top_n)
            return idx, dist


    def similarity_matrix(self, query_descriptors: dict, map_descriptors: dict) -> np.ndarray:
        return cosine_similarity(map_descriptors["map_descriptors"],
                                 query_descriptors["query_descriptors"]).astype(np.float32)


    def save_descriptors(self, dataset_name: str) -> None:
        if not os.path.isdir(netvlad_directory + "/descriptors/" + dataset_name):
            os.makedirs(netvlad_directory + "/descriptors/" + dataset_name)
        with open(netvlad_directory + "/descriptors/" + dataset_name + "/" + self.name + "_query.pkl", "wb") as f:
            pickle.dump(self.query_desc, f)
        with open(netvlad_directory + "/descriptors/" + dataset_name + "/" + self.name + "_map.pkl", "wb") as f:
            pickle.dump(self.map_desc, f)
        

    def load_descriptors(self, dataset_name: str) -> None:
        if not os.path.isdir(netvlad_directory + "/descriptors/" + dataset_name):
            raise Exception("Descriptor not yet computed for: " + dataset_name)
        with open(netvlad_directory + "/descriptors/" + dataset_name + "/" + self.name + "_query.pkl", "rb") as f:
            self.query_desc = pickle.load(f)
        with open(netvlad_directory + "/descriptors/" + dataset_name + "/" + self.name + "_map.pkl", "rb") as f:
            self.map_desc = pickle.load(f)
            self.set_map(self.map_desc)
        



