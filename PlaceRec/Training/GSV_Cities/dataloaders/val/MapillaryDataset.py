from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from PlaceRec.utils import get_config

config = get_config()

DATASET_ROOT = config["datasets_directory"] + "/mapillary_sls/"
GT_ROOT = DATASET_ROOT  # BECAREFUL, this is the ground truth that comes with GSV-Cities


class MSLS(Dataset):
    def __init__(self, input_transform=None):

        self.input_transform = input_transform

        self.dbImages = np.load(GT_ROOT + "msls_val_dbImages.npy")
        self.qIdx = np.load(GT_ROOT + "msls_val_qIdx.npy")
        self.qImages = np.load(GT_ROOT + "msls_val_qImages.npy")
        self.ground_truth = np.load(GT_ROOT + "msls_val_pIdx.npy", allow_pickle=True)

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
