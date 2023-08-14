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
    img_path = int(img_path.split('/')[-1][:-4])
    return img_path

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
            images += glob(section + '/*')
    images = np.array(images)
    image_index = [image_idx(img) for img in images]
    sort_idx = np.argsort(image_index)
    images = images[sort_idx]
    return images

    

class Nordlands(BaseDataset):
    def __init__(self):
        self.train_map_paths = get_paths("train", MAP_SET)
        self.train_query_paths = get_paths("train", QUERY_SET)
        self.test_map_paths = get_paths("test", MAP_SET)
        self.test_query_paths = get_paths("test", QUERY_SET)

        self.name = "norldlands"


    def query_images(self, partition: str, preprocess: torchvision.transforms.transforms.Compose = None) -> torch.Tensor:
        # get the required partition of the dataset
        if partition == "train": paths = self.train_query_paths[:int(len(self.train_query_paths)*0.8)]
        elif partition == "val": paths = self.train_query_paths[int(len(self.train_query_paths)*0.8):]
        elif partition == "test": paths = self.test_query_paths
        elif partition == "all": 
            paths = np.concatenate((self.train_query_paths, self.test_query_paths), axis=0)
            sort_index = [image_idx(img) for img in paths]
            sort_idx = np.argsort(sort_index)
            paths = paths[sort_idx]

        else: raise Exception("Partition must be 'train', 'val' or 'all'")
        
        if preprocess == None:
            return np.array([np.array(Image.open(pth)) for pth in paths])
        else: 
            imgs = np.array([np.array(Image.open(pth)) for pth in paths])
            return torch.stack([preprocess(q) for q in imgs])


    def map_images(self, partitions: str, preprocess: torchvision.transforms.transforms.Compose = None) -> torch.Tensor:
        if partition == "train" or "val":
            paths = self.train_map_paths
        elif partition == "test":
            paths = self.test_map_paths
        elif partition == "all":
            paths = np.concatenate((self.train_map_paths, self.test_map_paths), axis=0)
            sort_index = [image_idx(img) for img in images]
            sort_idx = np.argsort(sort_index)
            paths = paths[sort_idx]
        else: 
            raise Exception("Partition not found")

        if preprocess == None:
            return np.array([np.array(Image.open(pth)) for pth in self.map_paths])
        else: 
            imgs = np.array([np.array(Image.open(pth)) for pth in self.map_paths])
            return torch.stack([preprocess(q) for q in imgs])


    
    def query_images_loader(self, partition: str, batch_size: int = 16, shuffle: bool = False,
                            preprocess: torchvision.transforms.transforms.Compose = None, 
                            pin_memory: bool = False, 
                            num_workers: int = 0) -> torch.utils.data.DataLoader:


        # get the required partition of the dataset
        if partition == "train": paths = self.train_query_paths[:int(len(self.train_query_paths)*0.8)]
        elif partition == "val": paths = self.train_query_paths[int(len(self.train_query_paths)*0.8):]
        elif partition == "test": paths = self.test_query_paths
        elif partition == "all": 
            paths = np.concatenate((self.train_query_paths, self.test_query_paths), axis=0)
            sort_index = [image_idx(img) for img in paths]
            sort_idx = np.argsort(sort_index)
            paths = paths[sort_idx]
            
        else: raise Exception("Partition must be 'train', 'val' or 'all'")

        # build the dataloader
        dataset = ImageDataset(paths, preprocess=preprocess)
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, 
                                pin_memory=pin_memory, num_workers=num_workers)
        return dataloader


    def map_images_loader(self, partition: str, batch_size: int = 16, shuffle: bool = False,
                        preprocess: torchvision.transforms.transforms.Compose = None, 
                        pin_memory: bool = False, 
                        num_workers: int = 0) -> torch.utils.data.DataLoader:

        if partition == "train" or "val":
            paths = self.train_map_paths
        elif partition == "test":
            paths = self.test_map_paths
        elif partition == "all":
            paths = np.concatenate((self.train_map_paths, self.test_map_paths), axis=0)
            sort_index = [image_idx(img) for img in images]
            sort_idx = np.argsort(sort_index)
            paths = paths[sort_idx]
        else: 
            raise Exception("Partition not found")

        dataset = ImageDataset(self.map_paths, preprocess=preprocess)
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size,
                                pin_memory=pin_memory, num_workers=num_workers)
        return dataloader


    def ground_truth(self, partition: str, gt_type: str) -> np.ndarray:
        pass