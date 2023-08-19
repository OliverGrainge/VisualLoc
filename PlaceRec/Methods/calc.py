import torch 
import torchvision 
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import cv2
import random
import numpy as np
import torch.nn.functional as F
import os
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from glob import glob
from PIL import Image
from .base_method import BaseTechnique
from typing import Tuple
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sklearn
try:
    from ..utils import cosine_similarity_cuda
except: 
    pass

package_directory = os.path.dirname(os.path.abspath(__file__))




class CalcModel(nn.Module):
    def __init__(self,  pretrained=True, weights="/home/oliver/Documents/github/VisualLoc/PlaceRec/Methods/weights/calc.caffemodel.pt"):
        super().__init__()

        self.input_dim = (1, 120, 160)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5,5), stride=2, padding=4)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4,4), stride=1, padding=2)
        self.conv3 = nn.Conv2d(128, 4, kernel_size=(3,3), stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=(3,3), stride=2)
        self.lrn1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)
        self.lrn2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)

        if pretrained:
            state_dict = torch.load(weights)
            my_new_state_dict = {}
            my_layers = list(self.state_dict().keys())
            for layer in my_layers:
                my_new_state_dict[layer] = state_dict[layer]
            self.load_state_dict(my_new_state_dict)



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.lrn1(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.lrn2(x)

        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        return x


from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

class ConvertToYUVandEqualizeHist:
    def __call__(self, img):
        img_yuv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return Image.fromarray(img_rgb)



class CALC(BaseTechnique):
    
    def __init__(self):
        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = CalcModel(pretrained=True).to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            ConvertToYUVandEqualizeHist(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((120, 160), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            #transforms.Normalize((0.5,), (0.5,))  # Assuming the raw scale in Caffe transformation is equivalent to this normalization.
        ])



        self.map = None
        self.map_desc = None
        self.query_desc = None
        self.name = "calc"




    def compute_query_desc(self, images: torch.Tensor = None, dataloader: torch.utils.data.dataloader.DataLoader=None, pbar: bool=True) -> dict:
        if images is not None and dataloader is None:
            all_desc = self.model(images.to(self.device)).detach().cpu().numpy()
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(dataloader, desc="Computing CALC Query Desc", disable=not pbar):
                all_desc.append(self.model(batch.to(self.device)).detach().cpu().numpy())
            all_desc = np.vstack(all_desc)
        
        
        query_desc = {"query_descriptors": all_desc/np.linalg.norm(all_desc, axis=0)}
        self.set_query(query_desc)
        return query_desc


    def compute_map_desc(self, images: torch.Tensor = None, dataloader: torch.utils.data.dataloader.DataLoader=None, pbar: bool=True) -> dict:
        
        if images is not None and dataloader is None:
            all_desc = self.model(images.to(self.device)).detach().cpu().numpy()
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(dataloader, desc="Computing CALC Map Desc", disable=not pbar):
                all_desc.append(self.model(batch.to(self.device)).detach().cpu().numpy())
            all_desc = np.vstack(all_desc)
        else: 
            raise Exception("can only pass 'images' or 'dataloader'")

        map_desc = {'map_descriptors': all_desc/np.linalg.norm(all_desc, axis=0)}
        self.set_map(map_desc)

        return map_desc


    def set_query(self, query_descriptors: dict) -> None:
        self.query_desc = query_descriptors


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
        if not os.path.isdir(package_directory + "/descriptors/" + dataset_name):
            os.makedirs(package_directory + "/descriptors/" + dataset_name)
        with open(package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_query.pkl", "wb") as f:
            pickle.dump(self.query_desc, f)
        with open(package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_map.pkl", "wb") as f:
            pickle.dump(self.map_desc, f)
        

    def load_descriptors(self, dataset_name: str) -> None:
        if not os.path.isdir(package_directory + "/descriptors/" + dataset_name):
            raise Exception("Descriptor not yet computed for: " + dataset_name)
        with open(package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_query.pkl", "rb") as f:
            self.query_desc = pickle.load(f)
        with open(package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_map.pkl", "rb") as f:
            self.map_desc = pickle.load(f)
            self.set_map(self.map_desc)



