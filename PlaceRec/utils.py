import torch
import numpy as np
from torch.utils.data import Dataset 
from PIL import Image
import pathlib
import dropbox
from dropbox.exceptions import AuthError
from .db_key import DROPBOX_ACCESS_TOKEN


class ImageDataset(Dataset):
    def __init__(self, img_paths, preprocess=None):
        self.img_paths = img_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])

        if self.preprocess is not None:
            img = self.preprocess(img)

        return np.array(img)



def dropbox_connect():
    """Create a connection to Dropbox."""

    try:
        dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
    except AuthError as e:
        print('Error connecting to Dropbox with access token: ' + str(e))
    return dbx




def dropbox_download_file(dropbox_file_path, local_file_path):
    """Download a file from Dropbox to the local machine."""

    try:
        dbx = dropbox_connect()
        dbx.files_download_to_file(local_file_path, dropbox_file_path)

    except Exception as e:
        print('Error downloading file from Dropbox: ' + str(e))





def get_method(name: str=None):
    if name == "netvlad":
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
    elif name == "calc":
        from PlaceRec.Methods import CALC
        method = CALC()
    else:
        raise Exception("Method '" + name + "' not implemented")
    return method


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
    else:
        raise Exception("Dataset '" + name + "' not implemented")
    return dataset       