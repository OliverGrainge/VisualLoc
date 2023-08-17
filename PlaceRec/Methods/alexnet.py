import numpy as np
from tqdm import tqdm 
from .base_method import BaseTechnique
from typing import Tuple
import torch
from torchvision import transforms
from torchvision.models import AlexNet_Weights
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
import sklearn.neighbors
import pickle
import os
from tqdm import tqdm
try:
    from ..utils import cosine_similarity_cuda
except: 
    pass
# faiss only available on linux and mac
try:
    import faiss
except: 
    pass

alexnet_directory = os.path.dirname(os.path.abspath(__file__))

class AlexNet(BaseTechnique):
    
    def __init__(self):
        # Name of technique
        self.name = 'alexnet'

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=AlexNet_Weights.DEFAULT, verbose=False)
        self.model = self.model.features[:7]

        # send the model to relevant accelerator
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
        # Dimensions to project into
        self.nDims = 4096

        # elimminate gradient computations and send to accelerator
        self.model.to(self.device)
        self.model.eval()

        # preprocess for network 
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 244], antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.map = None
        self.map_desc = None
        self.query_desc = None

    def compute_query_desc(self, images: torch.Tensor=None, dataloader: torch.utils.data.dataloader.DataLoader=None, pbar: bool=True) -> dict:
        if images is not None and dataloader is None:
            desc = self.model(images.to(self.device)).detach().cpu().numpy()
            Ds = desc.reshape([images.shape[0], -1]) # flatten
            rng = np.random.default_rng(seed=0)
            Proj = rng.standard_normal([Ds.shape[1], self.nDims], 'float32')
            Proj = Proj / np.linalg.norm(Proj, axis=1, keepdims=True)
            Ds = Ds @ Proj
            all_desc = Ds
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(dataloader, desc="Computing AlexNet Query Desc", disable=not pbar):
                desc = self.model(batch.to(self.device)).detach().cpu().numpy()
                Ds = desc.reshape([batch.shape[0], -1]) # flatten
                rng = np.random.default_rng(seed=0)
                Proj = rng.standard_normal([Ds.shape[1], self.nDims], 'float32')
                Proj = Proj / np.linalg.norm(Proj, axis=1, keepdims=True)
                Ds = Ds @ Proj
                all_desc.append(Ds)
            all_desc = np.vstack(all_desc)
        else: 
            raise Exception("Can only pass 'images' or 'dataloader'")
        
    
        query_desc = {'query_descriptors': all_desc}
        self.set_query(query_desc)
        return query_desc


    def compute_map_desc(self, images: np.ndarray = None, dataloader: torch.utils.data.dataloader.DataLoader=None, pbar: bool=True) -> dict:
        if images is not None and dataloader is None:
            desc = self.model(images.to(self.device)).detach().cpu().numpy()
            Ds = desc.reshape([images.shape[0], -1]) # flatten
            rng = np.random.default_rng(seed=0)
            Proj = rng.standard_normal([Ds.shape[1], self.nDims], 'float32')
            Proj = Proj / np.linalg.norm(Proj, axis=1, keepdims=True)
            Ds = Ds @ Proj
            all_desc = Ds
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(dataloader, desc="Computing AlexNet Map Desc", disable=not pbar):
                desc = self.model(batch.to(self.device)).detach().cpu().numpy()
                Ds = desc.reshape([batch.shape[0], -1]) # flatten
                rng = np.random.default_rng(seed=0)
                Proj = rng.standard_normal([Ds.shape[1], self.nDims], 'float32')
                Proj = Proj / np.linalg.norm(Proj, axis=1, keepdims=True)
                Ds = Ds @ Proj
                all_desc.append(Ds)
            all_desc = np.vstack(all_desc)
        else: 
            raise Exception("Can only pass 'images' or 'dataloader'")
        
        map_desc = {'map_descriptors': all_desc}
        self.set_map(map_desc)
        return map_desc



    def set_map(self, map_descriptors: dict) -> None:
        self.map_desc = map_descriptors
        try: 
            # try to implement with faiss
            self.map = faiss.IndexFlatIP(map_descriptors["map_descriptors"].shape[1])
            faiss.normalize_L2(map_descriptors["map_descriptors"])
            self.map.add(map_descriptors["map_descriptors"])

        except: 
            # faiss is not available on unix or windows systems. In this case 
            # implement with scikit-learn
            from sklearn.neighbors import NearestNeighbors
            self.map = NearestNeighbors(n_neighbors=10, algorithm='auto', 
                                        metric='cosine').fit(map_descriptors["map_descriptors"])


    def set_query(self, query_descriptors: dict) -> None:
        self.query_desc = query_descriptors


    def place_recognise(self, images: torch.Tensor=None, dataloader: torch.utils.data.dataloader.DataLoader=None, top_n: int=1, pbar: bool=True) -> Tuple[np.ndarray, np.ndarray]:
        desc = self.compute_query_desc(images=images, dataloader=dataloader, pbar=pbar)
        if isinstance(self.map, sklearn.neighbors._unsupervised.NearestNeighbors):
            dist, idx = self.map.kneighbors(desc["query_descriptors"])
            return idx[:, :top_n], 1 - dist[:, :top_n]
        else: 
            faiss.normalize_L2(desc["query_descriptors"])
            dist, idx = self.map.search(desc["query_descriptors"], top_n)
            return idx, dist

    def similarity_matrix(self, query_descriptors: dict, map_descriptors: dict) -> np.ndarray:
        try:
            return cosine_similarity_cuda(map_descriptors["map_descriptors"], 
                                          query_descriptors["query_descriptors"]).astype(np.float32)
        except:
            return cosine_similarity(map_descriptors["map_descriptors"],
                                    query_descriptors["query_descriptors"]).astype(np.float32)

    def save_descriptors(self, dataset_name: str) -> None:
        if not os.path.isdir(alexnet_directory + "/descriptors/" + dataset_name):
            os.makedirs(alexnet_directory + "/descriptors/" + dataset_name)
        with open(alexnet_directory + "/descriptors/" + dataset_name + "/" + self.name + "_query.pkl", "wb") as f:
            pickle.dump(self.query_desc, f)
        with open(alexnet_directory + "/descriptors/" + dataset_name + "/" + self.name + "_map.pkl", "wb") as f:
            pickle.dump(self.map_desc, f)
        

    def load_descriptors(self, dataset_name: str) -> None:
        if not os.path.isdir(alexnet_directory + "/descriptors/" + dataset_name):
            raise Exception("Descriptor not yet computed for: " + dataset_name)
        with open(alexnet_directory + "/descriptors/" + dataset_name + "/" + self.name + "_query.pkl", "rb") as f:
            self.query_desc = pickle.load(f)
        with open(alexnet_directory + "/descriptors/" + dataset_name + "/" + self.name + "_map.pkl", "rb") as f:
            self.map_desc = pickle.load(f)
            self.set_map(self.map_desc)




        