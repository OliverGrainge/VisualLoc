import zipfile
import os
import numpy as np
from .base_dataset import BaseDataset
import torchvision
import torch
from glob import glob
from PIL import Image
from ..utils import ImageDataset, dropbox_download_file
from torch.utils.data import DataLoader
from scipy.signal import convolve2d


package_directory = os.path.dirname(os.path.abspath(__file__))

QUERY_SET = ["summer", "winter", "spring"]
MAP_SET = ["fall"]

def image_idx(img_path: str):
    img_path.split('/')[-1]
    print(img_path)

def get_paths(partition: list, seasons: list) -> list:
    if not os.path.isdir(package_directory + '/raw_images/Nordlands'): 
        print(package_directory + '/raw_images/nordlands')
        raise Exception("Please Download Nordlands dataset to /raw_images/Nordlands")
    
    root = package_directory + '/raw_images/Nordlands/' + partition
    images = []
    for season in seasons: 
        pth = root + '/' + season + '_images_' + partition
        sections = glob(pth + '/*')
        for section in sections: 
            print(section)
            images += glob(section + '/*')
    images = np.array(images)
    for img in images:
        image_idx(img)
    return 

    

class Nordlands(BaseDataset):
    def __init__(self):
        self.train_map_paths = get_paths("train", MAP_SET)
        self.train_query_paths = None 
        self.test_map_paths = None 
        self.test_query_patsh = None 

        self.name = "norldlands"


    def query_images(self, partition: str, preprocess: torchvision.transforms.transforms.Compose = None) -> torch.Tensor:
        pass

    def map_images(self, preprocess: torchvision.transforms.transforms.Compose = None) -> torch.Tensor:
        pass


    
    def query_images_loader(self, partition: str, batch_size: int = 16, shuffle: bool = False,
                            preprocess: torchvision.transforms.transforms.Compose = None, 
                            pin_memory: bool = False, 
                            num_workers: int = 0) -> torch.utils.data.DataLoader:
        pass


    def map_images_loader(self, batch_size: int = 16, shuffle: bool = False,
                        preprocess: torchvision.transforms.transforms.Compose = None, 
                        pin_memory: bool = False, 
                        num_workers: int = 0) -> torch.utils.data.DataLoader:
        pass


    def ground_truth(self, partition: str, gt_type: str) -> np.ndarray:
        pass