import os
import pickle
from os import path
from typing import Tuple

import cv2
import numpy as np
import sklearn
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from tqdm import tqdm

from .base_method import BaseFunctionality

hog_directory = os.path.dirname(os.path.abspath(__file__))


####################### PARAMETERS #########################
magic_width = 512
magic_height = 512
cell_size = 16  # HOG cell-size
bin_size = 8  # HOG bin size
image_frames = 1  # 1 for grayscale, 3 for RGB
descriptor_depth = bin_size * 4 * image_frames  # x4 is here for block normalization due to nature of HOG
ET = 0.4  # Entropy threshold, vary between 0-1.

total_no_of_regions = int((magic_width / cell_size - 1) * (magic_width / cell_size - 1))

#############################################################

#################### GLOBAL VARIABLES ######################

d1d2dot_matrix = np.zeros([total_no_of_regions, total_no_of_regions], dtype=np.float64)
d1d2matches_maxpooled = np.zeros([total_no_of_regions], dtype=np.float64)
d1d2matches_regionallyweighted = np.zeros([total_no_of_regions], dtype=np.float64)

matched_local_pairs = []
ref_desc = []


############################################################


def largest_indices_thresholded(ary):
    good_list = np.where(ary >= ET)
    #    no_of_good_regions=len(good_list[0])

    return good_list


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""

    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


# @jit(nopython=False)
def conv_match_dotproduct(d1, d2, regional_gd, total_no_of_regions):  # Assumed aspect 1:1 here
    global d1d2dot_matrix
    global d1d2matches_maxpooled
    global d1d2matches_regionallyweighted
    global matched_local_pairs

    np.dot(d1, d2, out=d1d2dot_matrix)

    np.max(d1d2dot_matrix, axis=1, out=d1d2matches_maxpooled)  # Select best matched ref region for every query region

    np.multiply(d1d2matches_maxpooled, regional_gd, out=d1d2matches_regionallyweighted)  # Weighting regional matches with regional goodness

    score = np.sum(d1d2matches_regionallyweighted) / np.sum(regional_gd)  # compute final match score

    return score


# ================================================== HOG ===============================================================


class HOG(BaseFunctionality):
    def __init__(self):
        self.device = "cpu"
        # preprocess images for method (None required)
        self.preprocess = transforms.ToTensor()
        # initialize the map to None
        self.map = None
        self.model = None
        # method name
        self.name = "hog"

        # Method Parameters
        winSize = (512, 512)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (16, 16)
        nbins = 9

        self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        self.winSize = winSize

        self.map_desc = None
        self.query_desc = None
        self.map = None

    def compute_query_desc(
        self,
        images: torch.Tensor = None,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
    ) -> dict:
        if images is not None and dataloader is None:
            images = list(np.array(images).transpose(0, 2, 3, 1))
            ref_desc_list = []
            for ref_image in images:
                if ref_image is not None:
                    img = ref_image * 255
                    img = cv2.cvtColor(
                        cv2.resize(img.astype(np.uint8), self.winSize),
                        cv2.COLOR_BGR2GRAY,
                    )
                    hog_desc = self.hog.compute(img)
                ref_desc_list.append(hog_desc)

            all_desc = np.array(ref_desc_list).astype(np.float32)
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(dataloader, desc="Computing HOG Query Desc", disable=not pbar):
                batch = list(np.array(batch).transpose(0, 2, 3, 1))
                ref_desc_list = []
                for ref_image in batch:
                    if ref_image is not None:
                        img = ref_image * 255
                        img = cv2.cvtColor(
                            cv2.resize(img.astype(np.uint8), self.winSize),
                            cv2.COLOR_BGR2GRAY,
                        )
                        hog_desc = self.hog.compute(img)
                    ref_desc_list.append(hog_desc)
                all_desc.append(np.array(ref_desc_list).astype(np.float32))
            all_desc = np.vstack(all_desc)

        query_desc = {"query_descriptors": all_desc}
        self.set_query(query_desc)
        return query_desc

    def compute_map_desc(
        self,
        images: torch.Tensor = None,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
    ) -> dict:
        if images is not None and dataloader is None:
            images = list(np.array(images).transpose(0, 2, 3, 1))
            ref_desc_list = []
            for ref_image in images:
                if ref_image is not None:
                    img = ref_image * 255
                    img = cv2.cvtColor(
                        cv2.resize(img.astype(np.uint8), self.winSize),
                        cv2.COLOR_BGR2GRAY,
                    )
                    hog_desc = self.hog.compute(img)
                ref_desc_list.append(hog_desc)

            all_desc = np.array(ref_desc_list).astype(np.float32)
        elif dataloader is not None and images is None:
            all_desc = []
            for batch in tqdm(dataloader, desc="Computing HOG Query Desc", disable=not pbar):
                batch = list(np.array(batch).transpose(0, 2, 3, 1))
                ref_desc_list = []
                for ref_image in batch:
                    if ref_image is not None:
                        img = ref_image * 255
                        img = cv2.cvtColor(
                            cv2.resize(img.astype(np.uint8), self.winSize),
                            cv2.COLOR_BGR2GRAY,
                        )
                        hog_desc = self.hog.compute(img)
                    ref_desc_list.append(hog_desc)
                all_desc.append(np.array(ref_desc_list).astype(np.float32))
            all_desc = np.vstack(all_desc)

        map_desc = {"map_descriptors": all_desc}
        self.set_map(map_desc)
        return map_desc
