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
from ..utils import get_dataset



package_directory = os.path.dirname(os.path.abspath(__file__))


class Combined(BaseDataset):

    def __init__(self, datasets=["essex3in1", "sfu", "nordlands", "gardenspointwalking", "stlucia_small"]):

        self.all_datasets = [get_dataset(name) for name in datasets]
        self.query_paths = np.concatenate([ds.query_partition("all") for ds in self.all_datasets])
        self.map_paths = np.concatenate([ds.map_partition("all") for ds in self.all_datasets])
        self.name = "combined"

    def query_partition(self, partition: str) -> np.ndarray:
        paths = np.concatenate([ds.query_partition(partition) for ds in self.all_datasets])
        return paths

    def map_partition(self, partition: str) -> np.ndarray:
        paths = np.concatenate([ds.map_partition(partition) for ds in self.all_datasets])
        return paths

    
    def query_images(self, partition: str, preprocess: torchvision.transforms.transforms.Compose = None) -> torch.Tensor:
        paths = self.query_partition(partition)
        if preprocess == None:
            return np.array([np.array(Image.open(pth))[:, :, :3] for pth in paths])
        else: 
            imgs = np.array([np.array(Image.open(pth))[:, :, :3] for pth in paths])
            return torch.stack([preprocess(q) for q in imgs])



    def map_images(self,partition: str, preprocess: torchvision.transforms.transforms.Compose = None) -> torch.Tensor:
        paths = self.map_partition(paths)
        if preprocess == None:
            return np.array([np.array(Image.open(pth)) for pth in paths])
        else: 
            imgs = np.array([np.array(Image.open(pth)) for pth in paths])
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

        # build the dataloader
        dataset = ImageDataset(paths, preprocess=preprocess)
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size,
                                pin_memory=pin_memory, num_workers=num_workers)
        return dataloader


    def ground_truth(self, partition: str, gt_type: str) -> np.ndarray:
        
        hard_ground_truths = [ds.ground_truth(partition=partition, gt_type='hard') for ds in self.all_datasets]
        soft_ground_truths = [ds.ground_truth(partition=partition, gt_type='soft') for ds in self.all_datasets]

        n_queries = np.sum([gt.shape[1] for gt in hard_ground_truths])
        n_refs = np.sum([gt.shape[0] for gt in hard_ground_truths])

        full_gt = np.zeros((n_refs, n_queries), dtype=bool)

        query_boundaries = [int(np.sum([hard_ground_truths[i].shape[1] for i in range(j)])) for j in range(len(self.all_datasets))]
        ref_boundaries = [int(np.sum([hard_ground_truths[i].shape[0] for i in range(j)])) for j in range(len(self.all_datasets))]

        if gt_type == "hard":
            for i in range(len(self.all_datasets)):
                startQ = query_boundaries[i]
                startR = ref_boundaries[i]
                if i+1 != len(self.all_datasets):
                    endQ = query_boundaries[i+1]
                    endR = ref_boundaries[i+1]
                    full_gt[startR:endR, startQ:endQ] = hard_ground_truths[i]
                else: 
                    full_gt[startR:, startQ:] = hard_ground_truths[i]

        elif gt_type == "soft":
            for i in range(len(self.all_datasets)):
                startQ = query_boundaries[i]
                startR = ref_boundaries[i]
                if i+1 != len(self.all_datasets):
                    endQ = query_boundaries[i+1]
                    endR = ref_boundaries[i+1]
                    full_gt[startR:endR, startQ:endQ] = soft_ground_truths[i]
                else: 
                    full_gt[startR:, startQ:] = soft_ground_truths[i]


        return full_gt.astype(bool)