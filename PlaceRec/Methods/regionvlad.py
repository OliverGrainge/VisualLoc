import torch
from torch import nn
import torch.nn.functional as F
import os
import itertools
from skimage.measure import regionprops, label
import numpy as np
from .base_method import BaseFunctionality
import pickle
from torchvision import transforms
from tqdm import tqdm
from typing import Tuple

package_directory = os.path.dirname(os.path.abspath(__file__))


def getROIs(imgConvFeat, imgLocalConvFeat, img):
    clustersEnergies_Ej = []
    clustersBoxes = []
    allROI_Box = []

    for featuremap in imgConvFeat:
        clusters = regionprops(
            label(featuremap), intensity_image=featuremap, cache=False
        )
        clustersBoxes.append(list(cluster.bbox for cluster in clusters))
        clustersEnergies_Ej.append(list(cluster.mean_intensity for cluster in clusters))
    # Make a list of ROIs with their bounded boxes
    clustersBoxes = list(itertools.chain.from_iterable(clustersBoxes))
    clustersEnergies_Ej = list(itertools.chain.from_iterable(clustersEnergies_Ej))
    # Sort the ROIs based on energies
    allROIs = sorted(clustersEnergies_Ej, reverse=True)
    # Pick up top N energetic ROIs with their bounding boxes
    allROIs = allROIs[:400]
    allROI_Box = [clustersBoxes[clustersEnergies_Ej.index(i)] for i in allROIs]
    #    clustersEnergies_Ej.clear()
    #    clustersBoxes.clear()
    aggregatedNROIs = np.zeros((400, imgLocalConvFeat.shape[2]))
    # Retreive the aggregated local descriptors lying under N ROIs
    for ROI in range(len(allROI_Box)):
        #      minRow, minCol, maxRow, maxCol = allROI_Box[ROI][0],allROI_Box[ROI][1],allROI_Box[ROI][2],allROI_Box[ROI][3]
        aggregatedNROIs[ROI, :] = np.sum(
            imgLocalConvFeat[
                allROI_Box[ROI][0] : allROI_Box[ROI][2],
                allROI_Box[ROI][1] : allROI_Box[ROI][3],
            ],
            axis=(0, 1),
        )  # Sometimes elements come out as nan for rare images, which breaks the code. I have picked this up with the author and for the time being patched this. Maybe some issue with package versions. Needs further debugging. #Mubariz

    # NxK dimensional ROIs
    return np.asarray(aggregatedNROIs)


# Retreive the VLAD representation
def getVLAD(X, visualDictionary):
    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels = visualDictionary.labels_
    k = visualDictionary.n_clusters

    m, d = X.shape
    Vlad = np.zeros([k, d])
    # computing the differences
    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels == i) > 0:
            Vlad[i] = np.sum(X[predictedLabels == i, :] - centers[i], axis=0)
    Vlad = Vlad.flatten()
    Vlad = np.sign(Vlad) * np.sqrt(np.abs(Vlad))
    Vlad = Vlad / np.sqrt(np.dot(Vlad, Vlad))
    Vlad = Vlad.reshape(k, d)
    return Vlad


class AlexnetPlaces365(nn.Module):
    def __init__(self):
        super().__init__()

        # conv1
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)

        # conv2
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)

        # conv3
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)

        # conv4
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2)

        # conv5
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        # fully connected
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(4096, 365)
        self.probs = nn.Softmax(dim=1)

    def forward(self, x):
        # conv 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)

        # conv 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.norm2(x)

        # conv3
        x = F.relu(self.conv3(x))
        return x


# ==========================================================================


class RegionVLAD(BaseFunctionality):
    def __init__(self, vocab_size: str = "small"):
        super().__init__()
        self.name = "regionvlad"

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = AlexnetPlaces365()
        self.model.load_state_dict(
            torch.load(package_directory + "/weights/alexnet_places365.caffemodel.pt")
        )
        self.model.eval()
        self.model.to(self.device)

        file = open(
            package_directory + "/weights/regionvlad_vocab_400.pkl",
            "rb",
        )
        self.vocab = pickle.load(file)
        file.close()

        self.mean_image = np.load(
            package_directory + "/weights/regionvlad_mean_image.npy"
        )

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((227, 227), antialias=True),
                transforms.Normalize(
                    mean=self.mean_image.mean(1).mean(1) / 255, std=[1, 1, 1]
                ),
                transforms.Lambda(lambda x: x * 255.0),
            ]
        )

    def desc_features(self, features, img):
        features = np.array(features)
        stacked_features = features.transpose(1, 2, 0)
        rois = getROIs(features, stacked_features, img)
        vocabulary = self.vocab["400"]["256"]["conv3"]
        vlad = getVLAD(rois, vocabulary)
        return vlad

    def compute_query_desc(
        self,
        images: torch.Tensor = None,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
    ) -> dict:
        if images is not None and dataloader is None:
            all_features = self.model(images.to(self.device)).detach().cpu().numpy()
            vlads = [
                self.desc_features(all_features[i], images[i])
                for i in range(images.shape[0])
            ]
            self.query_desc = {"query_descriptors": vlads}
            return self.query_desc

        elif dataloader is not None and images is None:
            vlads = []
            for batch in tqdm(
                dataloader, desc="Computing RegionVLAD Query Desc", disable=not pbar
            ):
                batch_features = (
                    self.model(batch.to(self.device)).detach().cpu().numpy()
                )
                vlads += [
                    self.desc_features(batch_features[i], batch[i])
                    for i in range(batch.shape[0])
                ]
            self.query_desc = {"query_descriptors": vlads}
            return self.query_desc
        else:
            raise Exception("Must either pass a Dataloader OR Images")

    def compute_map_desc(
        self,
        images: torch.Tensor = None,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
    ) -> dict:
        if images is not None and dataloader is None:
            all_features = self.model(images.to(self.device)).detach().cpu().numpy()
            vlads = [
                self.desc_features(all_features[i], images[i])
                for i in range(images.shape[0])
            ]
            self.map_desc = {"map_descriptors": vlads}
            return self.map_desc

        elif dataloader is not None and images is None:
            vlads = []
            for batch in tqdm(
                dataloader, desc="Computing RegionVLAD Map Desc", disable=not pbar
            ):
                batch_features = (
                    self.model(batch.to(self.device)).detach().cpu().numpy()
                )
                vlads += [
                    self.desc_features(batch_features[i], batch[i])
                    for i in range(batch.shape[0])
                ]
            self.map_desc = {"map_descriptors": vlads}
            return self.map_desc
        else:
            raise Exception("Must either pass a Dataloader OR Images")

    def similarity_matrix(
        self, query_descriptors: dict, map_descriptors: dict
    ) -> np.ndarray:
        similarity = np.zeros(
            (
                len(map_descriptors["map_descriptors"]),
                len(query_descriptors["query_descriptors"]),
            ),
            dtype=np.float32,
        )
        for i in range(similarity.shape[0]):
            for j in range(similarity.shape[1]):
                desc1 = map_descriptors["map_descriptors"][i]
                desc2 = query_descriptors["query_descriptors"][j]
                similarity[i, j] = np.sum(np.einsum("ij,ij->i", desc1, desc2))
        return similarity

    def set_map(self, map_descriptors: dict) -> None:
        self.map_desc = map_descriptors
        self.map = map_descriptors

    def place_recognise(
        self,
        images: torch.Tensor = None,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        top_n: int = 1,
        pbar: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # this is a single threaded brute force approach
        # will be very slow
        if self.map_desc is None:
            raise Exception(
                "Please Compute the map features with 'compute_map_desc' before performing vpr"
            )

        if images is not None and dataloader is None:
            q_desc = self.compute_query_desc(images=images)
        elif dataloader is not None and images is None:
            q_desc = self.compute_query_desc(dataloader=dataloader)
        else:
            raise Exception("Can only pass images OR dataloader")

        scores = np.zeros(
            (len(self.map_desc["map_descriptors"]), len(q_desc["query_descriptors"]))
        )
        for i in range(len(self.map_desc["map_descriptors"])):
            for j in range(len(q_desc["query_descriptors"])):
                scores[i, j] = np.sum(
                    np.einsum(
                        "ij,ij->i",
                        q_desc["query_descriptors"][j],
                        self.map_desc["map_descriptors"][i],
                    )
                )

        idxs = scores.argsort(0)
        idxs = idxs[:top_n, :].transpose()
        dists = scores[
            idxs,
            np.repeat(np.arange(top_n).astype(int)[None, :], idxs.shape[0], axis=0),
        ]
        return idxs, dists.astype(np.float32)
