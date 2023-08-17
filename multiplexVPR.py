from PlaceRec.Methods.base_method import BaseTechnique
import torch
from PlaceRec.utils import get_method, ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sklearn 
from sklearn.neighbors import NearestNeighbors
import os
from sklearn.metrics.pairwise import cosine_similarity
from train_selection import ResNet18, select_transform
from typing import Tuple
import pickle

METHODS = ["mixvpr", "netvlad", "cosplace", "convap"]
SELECTION_MODEL_PATH = "/home/oliver/Documents/github/VisualLoc/SelectionNetworkCheckpoints/combined_largesets_recall@1_resnet18-epoch=106-val_loss=0.55.ckpt"
BATCHSIZE = 160


package_directory = os.path.dirname(os.path.abspath(__file__))


class MultiPlexVPR(BaseTechnique):

    def __init__(self):
    
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
        self.name = "multiplexvpr"
        self.map = None
        self.map_desc = None
        self.query_desc = None 
        self.methods = [get_method(method_name) for method_name in METHODS]
        self.selection_model = ResNet18.load_from_checkpoint(SELECTION_MODEL_PATH, output_dim=len(METHODS))
        self.preprocess =  select_transform

    def compute_selections(self, images: torch.Tensor = None, dataloader: torch.utils.data.dataloader.DataLoader=None, pbar: bool=True) -> torch.Tensor:
        if images is not None and dataloader is None:
            probs = self.selection_model(images.to(self.device)).detach().cpu()
            selections = torch.argmax(probs, axis=1)
            return selections.type(torch.int)
        elif dataloader is not None and images is None: 
            selections = []
            for batch in dataloader:
                probs = self.selection_model(batch.to(self.device)).detach().cpu()
                sel = torch.argmax(probs, axis=1).numpy().tolist()
                selections += sel
            return torch.Tensor(selections).type(torch.int)
        else: 
            raise Exception("Can only pass a dataloader or images, not both")


    def compute_query_desc(self, images: torch.Tensor = None, dataloader: torch.utils.data.dataloader.DataLoader = None, pbar: bool=True) -> dict:
        if images is not None and dataloader is not None:
            raise NotImplementedError
        elif dataloader is not None and images is None: 
            paths = dataloader.dataset.img_paths
            progress_bar  = tqdm(total=len(paths), desc="Computing MultiPlexVPR Query Descriptors", leave=True, disable= not pbar)
            selections = self.compute_selections(dataloader=dataloader, images=None).numpy()
            select_masks = [np.where(np.array(selections)==i) for i in range(len(METHODS))]
            select_queries = [np.array(paths)[mask] for mask in select_masks]

            all_desc = []
            for i, queries in enumerate(select_queries):
                if len(queries) >= 1:
                    ds = ImageDataset(img_paths=queries, preprocess=self.methods[i].preprocess)
                    dl = DataLoader(ds, shuffle=False, batch_size=BATCHSIZE)
                    progress_bar.update(ds.__len__())
                    desc = self.methods[i].compute_query_desc(dataloader=dl, pbar=False)["query_descriptors"]
                    all_desc.append(desc)
                else: 
                    all_desc.append(np.array([]))

            q_desc = []
            counters = np.zeros(len(METHODS), dtype=int)
            for i, sel in enumerate(selections):
                q_desc.append(all_desc[sel][counters[sel]])
                counters[sel] += 1

            all_desc = {"query_descriptors": q_desc, "selections": selections}
            self.set_query(all_desc)
            return all_desc

            

    def compute_map_desc(self, images: torch.Tensor = None, dataloader: torch.utils.data.dataloader.DataLoader = None, pbar: bool=True) -> dict:
        if images is not None and dataloader is not None:
            raise NotImplementedError
        elif dataloader is not None and images is None:
            img_paths = dataloader.dataset.img_paths
            progress_bar  = tqdm(total=dataloader.dataset.__len__()*len(METHODS), desc="Computing MultiPlexVPR Map Descriptors", leave=True, disable= not pbar)
            all_desc = []
            for method in self.methods:

                ds = ImageDataset(img_paths=img_paths, preprocess=method.preprocess)
                dl = DataLoader(ds, shuffle=False, batch_size=BATCHSIZE)
                all_desc.append(method.compute_map_desc(dataloader=dl, pbar=False)["map_descriptors"])
                progress_bar.update(ds.__len__())
            map_desc = {"map_descriptors": all_desc}
            self.set_map(map_desc)
            return map_desc
     
    def set_map(self, map_descriptors: dict) -> None:
        self.map_desc = map_descriptors
        all_desc = map_descriptors["map_descriptors"]
        self.map = []
        for desc in all_desc:
            self.map_desc = map_descriptors
            try: 
                map_idx = faiss.IndexFlatIP(desc.shape[1])
                faiss.normalize_L2(desc)
                map_idx.add(desc)
                self.map.append(map_idx)

            except: 
                map_idx = NearestNeighbors(n_neighbors=10, algorithm='auto', 
                                            metric='cosine').fit(desc)
                self.map.append(map_idx)

    def set_query(self, query_descriptors: dict) -> None:
        self.query_desc = query_descriptors

    def place_recognise(self, images: torch.Tensor = None, dataloader: torch.utils.data.dataloader.DataLoader = None, pbar: bool=True, top_n: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def similarity_matrix(self, query_descriptors: np.ndarray, map_descriptors: np.ndarray) -> np.ndarray:
        q_desc = query_descriptors["query_descriptors"]
        selections = query_descriptors["selections"]
        masks = [np.where(np.array(selections) == i) for i in range(len(METHODS))]
        select_descriptors = [np.array(q_desc, dtype=object)[mask] for mask in masks]
        select_descriptors = [np.vstack(d) if len(d) > 0 else np.array([]) for d in select_descriptors]
        m_desc = map_descriptors["map_descriptors"]
        sep_S = [cosine_similarity(d, m_desc[i]) if len(d) != 0 else np.zeros(m_desc[0].shape[0]) for i, d in enumerate(select_descriptors)]
        S = np.zeros((len(selections), m_desc[0].shape[0]))
        for i, mask in enumerate(masks):
            S[mask] = sep_S[i]
        S = S.transpose()
        return S








    def save_descriptors(self, dataset_name: str) -> None:
        if not os.path.isdir(package_directory + "/descriptors/" + dataset_name):
            os.makedirs(package_directory + "/descriptors/" + dataset_name)
        with open(package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_query.pkl", "wb") as f_q:
            pickle.dump(self.query_desc, f_q)
        with open(package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_map.pkl", "wb") as f_m:
            pickle.dump(self.map_desc, f_m)
        

    def load_descriptors(self, dataset_name: str) -> None:
        if not os.path.isdir(package_directory + "/descriptors/" + dataset_name):
            raise Exception("Descriptor not yet computed for: " + dataset_name)
        with open(package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_query.pkl", "rb") as f_q:
            self.query_desc = pickle.load(f_q)
        with open(package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_map.pkl", "rb") as f_m:
            self.map_desc = pickle.load(f_m)
        self.set_map(self.map_desc)