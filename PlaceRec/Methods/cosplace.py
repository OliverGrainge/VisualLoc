from .base_method import BaseTechnique
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

try:
    from ..utils import cosine_similarity_cuda
except: 
    pass

try:
    import faiss
except: 
    pass


cosplace_directory = os.path.dirname(os.path.abspath(__file__))


class CosPlace(BaseTechnique):

    def __init__(self):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")


        self.model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet50", fc_output_dim=2048, 
                                    verbose=False).to(self.device)
        self.model.eval()


        # preprocess images for method (None required)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # initialize the map to None
        self.map = None
        self.map_desc = None
        self.query_desc = None
        # method name
        self.name = "cosplace"


    def compute_query_desc(self, images: torch.Tensor = None, dataloader = None, pbar: bool=True) -> dict:
        if images is not None and dataloader is None:
            with torch.no_grad():
                all_desc = self.model(images.to(self.device)).detach().cpu().numpy()
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(dataloader, desc="Computing CosPlace Query Desc", disable=not pbar):
                with torch.no_grad()
                    all_desc.append(self.model(batch.to(self.device)).detach().cpu().numpy())
            all_desc = np.vstack(all_desc)
        
        
        query_desc = {"query_descriptors": all_desc}
        self.set_query(query_desc)
        return query_desc

    def compute_map_desc(self, images: torch.Tensor = None, dataloader=None, pbar: bool=True) -> dict:

        if images is not None and dataloader is None:
            with torch.no_grad():
                all_desc = self.model(images.to(self.device)).detach().cpu().numpy()
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(dataloader, desc="Computing CosPlace Map Desc", disable=not pbar):
                with torch.no_grad()
                    all_desc.append(self.model(batch.to(self.device)).detach().cpu().numpy())
            all_desc = np.vstack(all_desc)
        else: 
            raise Exception("can only pass 'images' or 'dataloader'")

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
            self.map = NearestNeighbors(n_neighbors=10, algorithm='auto', 
                                        metric='cosine').fit(map_descriptors["map_descriptors"])

    def set_query(self, query_descriptors: dict) -> None:
        self.query_desc = query_descriptors


    def place_recognise(self, images: torch.Tensor=None, dataloader: torch.utils.data.dataloader.DataLoader = None, top_n: int=1, pbar: bool=True) -> Tuple[np.ndarray, np.ndarray]:
            desc = self.compute_query_desc(images=images, dataloader=dataloader, pbar=pbar)
            if isinstance(self.map, sklearn.neighbors._unsupervised.NearestNeighbors):
                dist, idx = self.map.kneighbors(desc["query_descriptors"])
                return idx[:, :top_n], 1 - dist[:, :top_n]
            else: 
                faiss.normalize_L2(desc["query_descriptors"])
                dist, idx = self.map.search(desc["query_descriptors"], top_n)
                return idx, dist
            

    def similarity_matrix(self, query_descriptors: dict, map_descriptors: dict) -> np.ndarray:
        if self.device == 'cuda': 
            return cosine_similarity_cuda(map_descriptors["map_descriptors"], 
                                          query_descriptors["query_descriptors"]).astype(np.float32)
        else: 
            return cosine_similarity(map_descriptors["map_descriptors"],
                                    query_descriptors["query_descriptors"]).astype(np.float32)

    
    def save_descriptors(self, dataset_name: str) -> None:
        if not os.path.isdir(cosplace_directory + "/descriptors/" + dataset_name):
            os.makedirs(cosplace_directory + "/descriptors/" + dataset_name)
        with open(cosplace_directory + "/descriptors/" + dataset_name + "/" + self.name + "_query.pkl", "wb") as f:
            pickle.dump(self.query_desc, f)
        with open(cosplace_directory + "/descriptors/" + dataset_name + "/" + self.name + "_map.pkl", "wb") as f:
            pickle.dump(self.map_desc, f)
        

    def load_descriptors(self, dataset_name: str) -> None:
        if not os.path.isdir(cosplace_directory + "/descriptors/" + dataset_name):
            raise Exception("Descriptor not yet computed for: " + dataset_name)
        with open(cosplace_directory + "/descriptors/" + dataset_name + "/" + self.name + "_query.pkl", "rb") as f:
            self.query_desc = pickle.load(f)
        with open(cosplace_directory + "/descriptors/" + dataset_name + "/" + self.name + "_map.pkl", "rb") as f:
            self.map_desc = pickle.load(f)
            self.set_map(self.map_desc)



        