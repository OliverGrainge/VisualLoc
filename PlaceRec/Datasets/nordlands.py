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
from tqdm import tqdm
from scipy.signal import convolve2d

package_directory = os.path.dirname(os.path.abspath(__file__))

QUERY_SET = ["summer"]
MAP_SET = ["fall"]

def image_idx(img_path: str):
    img_path = int(img_path.split('/')[-1][:-4])
    return img_path

def get_paths(partition: list, seasons: list) -> list:
    if not os.path.isdir(package_directory + '/raw_images/Nordlands'): 
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

        self.query_paths = self.query_partition("all")
        self.map_paths = self.map_partition("all")

        self.name = "norldlands"

    def query_partition(self, partition: str) -> np.ndarray:
        # get the required partition of the dataset
        if partition == "train": paths = self.train_query_paths[:int(len(self.train_query_paths)*0.8)]
        elif partition == "val": paths = self.train_query_paths[int(len(self.train_query_paths)*0.8):]
        elif partition == "test": paths = self.test_query_paths
        elif partition == "all": paths = self.train_query_paths
        else: raise Exception("Partition must be 'train', 'val' or 'all'")
        return paths

    def map_partition(self, partition: str) -> np.ndarray:
        if partition == "train" or partition == "val": 
            paths = self.train_map_paths
        elif partition == "test": 
            paths = self.test_map_paths
        elif partition == "all": 
            paths = self.train_map_paths
        else: 
            raise Exception("Partition not found")
        return paths


    def query_images(self, partition: str, preprocess: torchvision.transforms.transforms.Compose = None) -> torch.Tensor:
        
        paths = self.query_partition(partition)
        
        if preprocess == None:
            return np.array([np.array(Image.open(pth).resize((320, 320)))[:, :, :3] for pth in paths])
        else: 
            imgs = np.array([np.array(Image.open(pth).resize((320, 320)))[:, :, :3] for pth in paths])
            return torch.stack([preprocess(q) for q in imgs])


    def map_images(self, partition: str, preprocess: torchvision.transforms.transforms.Compose = None) -> torch.Tensor:
        paths = self.map_partition(partition)

        if preprocess == None:
            return np.array([np.array(Image.open(pth).resize((320, 320)))[:, :, :3] for pth in paths])
        else: 
            imgs = np.array([np.array(Image.open(pth).resize((320, 320)))[:, :, :3] for pth in paths])
            return torch.stack([preprocess(q) for q in imgs])


    
    def query_images_loader(self, partition: str, batch_size: int = 16, shuffle: bool = False,
                            preprocess: torchvision.transforms.transforms.Compose = None, 
                            pin_memory: bool = False, 
                            num_workers: int = 0) -> torch.utils.data.DataLoader:


        paths = self.query_partition(partition)

        # build the dataloader
        dataset = ImageDataset(paths, preprocess=preprocess)
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, 
                                pin_memory=pin_memory, num_workers=num_workers)
        return dataloader


    def map_images_loader(self, partition: str, batch_size: int = 16, shuffle: bool = False,
                        preprocess: torchvision.transforms.transforms.Compose = None, 
                        pin_memory: bool = False, 
                        num_workers: int = 0) -> torch.utils.data.DataLoader:
        
        paths = self.map_partition(partition)
        dataset = ImageDataset(paths, preprocess=preprocess)
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size,
                                pin_memory=pin_memory, num_workers=num_workers)
        return dataloader


    def ground_truth(self, partition: str, gt_type: str) -> np.ndarray:
        query_paths = self.query_partition(partition)
        map_paths = self.map_partition(partition)
        query_idx = np.array([image_idx(img) for img in query_paths])
        map_idx = np.array([image_idx(img) for img in map_paths])
        ground_truth = map_idx[:, np.newaxis] == query_idx

        # this if statement is very slow for the huge ground truth!
        # maybe comment out if trying to run tests fast.
        if gt_type == "soft":
            ground_truth = convolve2d(ground_truth.astype(int), np.ones((15, 1), 'int'), mode='same').astype('bool')
        return ground_truth


        