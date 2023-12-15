from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from os.path import join
from PlaceRec.utils import get_config
import os


config = get_config()
DATASET_ROOT = join(config["train"]["datasets_folder"], "gsv_cities", "validation_data", "msls_val")
GT_ROOT = os.path.dirname(os.path.abspath(__file__)).replace("dataloaders/val", "datasets/")


class MSLS(Dataset):
    def __init__(self, input_transform = None):
        

        self.input_transform = input_transform

        self.dbImages = np.load(GT_ROOT+'msls_val/msls_val_dbImages.npy')
        self.qIdx = np.load(GT_ROOT+'msls_val/msls_val_qIdx.npy')
        self.qImages = np.load(GT_ROOT+'msls_val/msls_val_qImages.npy')
        self.ground_truth = np.load(GT_ROOT+'msls_val/msls_val_pIdx.npy', allow_pickle=True)
        
        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages[self.qIdx])
    
    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT + self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)