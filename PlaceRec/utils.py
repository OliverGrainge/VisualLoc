import torch
import numpy as np
from torch.utils.data import Dataset 
from PIL import Image
import pathlib
import dropbox
from dropbox.exceptions import AuthError
from .db_key import DROPBOX_ACCESS_TOKEN
import torch
try: 
    import cupy as cp

    def cosine_similarity_cuda(A, B):
        # Normalize the rows of A and B
        A_norm = A / cp.linalg.norm(A, axis=1, keepdims=True)
        B_norm = B / cp.linalg.norm(B, axis=1, keepdims=True)
        # Compute the dot product between normalized A and B
        similarity = cp.dot(A_norm, B_norm.T)
        return similarity
except: 
    pass


class ImageDataset(Dataset):
    def __init__(self, img_paths, preprocess=None):
        self.img_paths = img_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        

        if self.preprocess is not None:
            img = np.array(Image.open(self.img_paths[idx]).resize((320, 320)))[:, :, :3]
            img = Image.fromarray(img)
            img = self.preprocess(img)
            return img

        img = np.array(Image.open(self.img_paths[idx]).resize((320, 320)))[:, :, :3]
        return img


def dropbox_connect():
    """Create a connection to Dropbox."""

    try:
        dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
    except AuthError as e:
        print('Error connecting to Dropbox with access token: ' + str(e))
    return dbx



def dropbox_download_file(dropbox_file_path, local_file_path):

    try:
        dbx = dropbox_connect()
        dbx.files_download_to_file(local_file_path, dropbox_file_path)


    except Exception as e:
        print('Error downloading file from Dropbox: ' + str(e))


 


def get_dataset(name: str=None):
    if name == "gardenspointwalking":
        from PlaceRec.Datasets import GardensPointWalking
        dataset = GardensPointWalking()
    elif name == "stlucia_small":
        from PlaceRec.Datasets import StLucia_small
        dataset = StLucia_small()
    elif name == "sfu":
        from PlaceRec.Datasets import SFU
        dataset = SFU()
    elif name == "essex3in1":
        from PlaceRec.Datasets import ESSEX3IN1
        dataset = ESSEX3IN1()
    elif name == "nordlands":
        from PlaceRec.Datasets import Nordlands
        dataset = Nordlands()
    elif name == "gsvcities":
        from PlaceRec.Datasets import GsvCities
        dataset = GsvCities()
    elif name == "combined":
        from PlaceRec.Datasets import Combined
        dataset = Combined()
    else:
        raise Exception("Dataset '" + name + "' not implemented")
    return dataset      


def get_method(name: str=None):
    if name == "calc":
        from PlaceRec.Methods import CALC
        method = CALC()
    elif name == "netvlad":
        from PlaceRec.Methods import NetVLAD
        method = NetVLAD()
    elif name == "hog": 
        from PlaceRec.Methods import HOG
        method = HOG()
    elif name == "cosplace":
        from PlaceRec.Methods import CosPlace
        method = CosPlace()
    elif name == "alexnet":
        from PlaceRec.Methods import AlexNet
        method = AlexNet()
    elif name == "mixvpr":
        from PlaceRec.Methods import MixVPR
        method = MixVPR()
    elif name == "convap":
        from PlaceRec.Methods import CONVAP
        method = CONVAP()
    else: 
        raise Exception("Method not implemented")
    return method
