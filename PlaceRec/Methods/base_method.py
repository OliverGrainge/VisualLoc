import os
import pickle
from abc import ABC, abstractmethod
from typing import Tuple

import faiss
import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from PlaceRec.utils import get_config

package_directory = os.path.dirname(os.path.abspath(__file__))

config = get_config()


class BaseTechnique(ABC):
    """
    This is an abstract class that serves as a template for visual place recognition
    technique implementations. All abstract methods must be implemented in each technique.

    Attributes:
        map (fiass.index or sklearn.neighbors.NearestNeighbors): this is a structure consisting of all descriptors computed with
                           "compute_map_desc". It is search by "place_recognise" when
                           performing place recognition
    """

    map = None

    @abstractmethod
    def compute_query_desc(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
    ) -> dict:
        """
        computes the image descriptors of queries and returns them as a numpy array

        Args:
            query_images (np.ndarray): Images as a numpy array in (H, W, C) representation with a uint8 data type

        Returns:
            dict: a dict describing the images. The contents of which will be determined by the particular method used.
        """
        pass

    @abstractmethod
    def compute_map_desc(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
    ) -> dict:
        """
               computes the image descriptors of the map and returns them as a dictionary. The particular format
               of the dictionary will depend on the type of vpr technique used

               Args:
                   map_images (np.ndarray): Images as a numpy array in (H, W, C) representation with a uint8 data type

        Returns:
                   dict: a dict describing the images. The contents of which will be determined by the particular method used.
        """
        pass

    @abstractmethod
    def set_map(self, map_descriptors: dict) -> None:
        """
        Sets the map attribute of this class with the map descriptors computed with "compute_map_desc". This
        map is searched by "place_recognise" to perform place recognition.

        Args:
            map (dict): dict: a dict describing the images. The contents of which will
                             be determined by the particular description method used.

        Returns:
            None:
        """
        pass

    @abstractmethod
    def set_query(self, query_descriptors: dict) -> None:
        """
        Sets the query_descriptor of the class.

        Args:
            query_descriptors (dict): a dictionary of query descriptors produced by the "compute_query_desc" function

        Returns:
            None
        """
        pass

    @abstractmethod
    def place_recognise(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
        top_n: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs place recognition by computing query image representations and using them to search the map attribute
        to find the relevant map

        Args:
            queries (np.ndarray): Images as a numpy array in (H, W, C) representation with a uint8 data type

            top_n (int): Determines the top N places to match and return.

        Returns:
            Tuple[np.ndarray, np.ndarray]: a tuple with the first np.ndarray being a array of [N, H, W, C]
            images that match the query place. the second np.ndarray is a matrix of similarities of size
            [N, top_n] measuring the cosine similarity between the query images and returned places

        """
        pass

    @abstractmethod
    def similarity_matrix(self, query_descriptors: np.ndarray, map_descriptors: np.ndarray) -> np.ndarray:
        """
        computes the similarity matrix using the cosine similarity metric. It returns
        a numpy matrix M. where M[i, j] determines how similar query i is to map image j.

        Args:
            query_descriptors (dict): dict of query descriptors computed by "compute_query_desc"
            map_descriptors (dict): dict of query descriptors computed by "compute_map_desc"

        Returns:
            np.ndarray: matrix M where M[i, j] measures cosine similarity between query image i and map image j
        """
        pass

    @abstractmethod
    def save_descriptors(dataset_name: str) -> None:
        """
        Method saves both the query and map attributes of the class to disk.

        Args:
            dataset_name (str): The name of the dataset on which the descriptors were computed:

        Returns:
            None
        """
        pass

    @abstractmethod
    def load_descriptors(dataset_name: str) -> None:
        """
        Loads the descriptors computed on the "dataset_name" from disk

        Args:
            dataset_name (str): The name of the dataset to load the descriptors from
        """
        pass


class BaseFunctionality(BaseTechnique):
    """
    This class provides the basic functionality for place recognition tasks.
    It allows setting and querying of descriptors, saving and loading of descriptors,
    and computing a similarity matrix.
    """

    def __init__(self):
        super().__init__()
        """
        Initialize the BaseFunctionality class.
        
        Attributes:
            query_desc: A dictionary containing query descriptors.
            map_desc: A dictionary containing map descriptors.
            map: A FAISS index for map descriptors.
            name: A string representing the name of the method.
            device: A string representing the compute device (cuda, mps, or cpu).
            model: A pytorch model for feature extraction
        """
        self.query_desc = None
        self.map_desc = None
        self.map = None
        self.name = None
        self.model = None

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def set_query(self, query_descriptors: dict) -> None:
        """
        Set the query descriptors.

        Args:
            query_descriptors (dict): A dictionary containing query descriptors.
        """
        self.query_desc = query_descriptors

    def set_map(self, map_descriptors: dict) -> None:
        """
        Set the map descriptors and initialize a FAISS index for them.

        Args:
            map_descriptors (dict): A dictionary containing map descriptors.
        """
        self.map_desc = map_descriptors
        if config["eval"]["distance"] == "cosine":
            self.map = faiss.IndexFlatIP(map_descriptors["map_descriptors"].shape[1])
            faiss.normalize_L2(map_descriptors["map_descriptors"])
            self.map.add(map_descriptors["map_descriptors"])
        elif config["eval"]["distance"] == "l2":
            self.map = faiss.IndexFlatL2(map_descriptors["map_descriptors"].shape[1])
            self.map.add(map_descriptors["map_descriptors"])
        else: 
            raise NotImplementedError("Distance Measure Not Implemented")

    def place_recognise(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        query_desc: np.ndarray = None,
        k: int = 1,
        pbar: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recognize places based on images or a dataloader.

        Args:
            images (torch.Tensor): A batch of images.
            dataloader (torch.utils.data.dataloader.DataLoader): A DataLoader for images.
            top_n (int): Number of top results to return.
            pbar (bool): Whether to show a progress bar.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Indices and distances of recognized places.
        """
        if query_desc == None and dataloader is not None:
            query_desc = self.compute_query_desc(dataloader=dataloader, pbar=pbar)
        if config["eval"]["distance"] == "cosine":
            faiss.normalize_L2(query_desc["query_descriptors"])
        dist, idx = self.map.search(query_desc["query_descriptors"], k)
        return idx, dist

    def similarity_matrix(self, query_descriptors: dict, map_descriptors: dict) -> np.ndarray:
        """
        Compute the similarity matrix between query and map descriptors.

        Args:
            query_descriptors (dict): A dictionary containing query descriptors.
            map_descriptors (dict): A dictionary containing map descriptors.

        Returns:
            np.ndarray: A similarity matrix.
        """
        return cosine_similarity(map_descriptors["map_descriptors"], query_descriptors["query_descriptors"]).astype(np.float32)

    def save_descriptors(self, dataset_name: str) -> None:
        """
        Save the descriptors to disk.

        Args:
            dataset_name (str): Name of the dataset for which descriptors are saved.
        """
        if not os.path.isdir(package_directory + "/descriptors/" + dataset_name):
            os.makedirs(package_directory + "/descriptors/" + dataset_name)
        with open(
            package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_query.pkl",
            "wb",
        ) as f:
            pickle.dump(self.query_desc, f)
        with open(
            package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_map.pkl",
            "wb",
        ) as f:
            pickle.dump(self.map_desc, f)

    def load_descriptors(self, dataset_name: str) -> None:
        """
        Load the descriptors from disk.

        Args:
            dataset_name (str): Name of the dataset for which descriptors are loaded.

        Raises:
            Exception: If descriptors for the given dataset are not found.
        """
        if not os.path.isdir(package_directory + "/descriptors/" + dataset_name):
            raise Exception("Descriptor not yet computed for: " + dataset_name)
        with open(
            package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_query.pkl",
            "rb",
        ) as f:
            self.query_desc = pickle.load(f)
        with open(
            package_directory + "/descriptors/" + dataset_name + "/" + self.name + "_map.pkl",
            "rb",
        ) as f:
            self.map_desc = pickle.load(f)
            self.set_map(self.map_desc)

    def set_device(self, device: str) -> None:
        """
        Set the device for the model.

        This method assigns the specified device to the model instance
        and moves the model to that device.

        Args:
            device (str): The device to which the model should be moved.
                        Common values are 'cpu', 'cuda:0', etc.

        Returns:
            None
        """
        self.device = device
        self.model.to(device)


class BaseModelWrapper(BaseFunctionality):
    """
    A wrapper for models that provides methods to compute query and map descriptors.

    This class inherits from `BaseFunctionality` and provides an interface
    to set up a model, preprocess its inputs, and compute descriptors for given data.

    Attributes:
        name (str): A name or identifier for the model.
        model (torch.nn.Module): The PyTorch model instance.
        preprocess (callable): A function or callable to preprocess input data.
        device (str): The device on which the model runs (inherited from `BaseFunctionality`).
    """

    def __init__(self, model, preprocess, name):
        """
        Initializes a BaseModelWrapper instance.

        Args:
            model (torch.nn.Module): The PyTorch model to be wrapped.
            preprocess (callable): A function or callable to preprocess the input data.
            name (str): A name or identifier for the model.
        """

        super().__init__()
        self.name = name
        self.model = model
        self.preprocess = preprocess
        self.model.to(self.device)
        self.model.eval()
        self.features_dim = self.features_size()

    def features_size(self):
        img = np.random.rand(224, 224, 3) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img = self.preprocess(img)
        with torch.no_grad():
            features = self.model(img[None, :].to(self.device)).detach().cpu()
        return features.size(1)

    def compute_query_desc(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
    ) -> dict:
        """
        Compute the query descriptors for the given data.

        Args:
            dataloader (torch.utils.data.dataloader.DataLoader, optional): DataLoader providing the data.
            pbar (bool, optional): If True, display a progress bar. Defaults to True.

        Returns:
            dict: A dictionary containing the computed query descriptors.
        """

        all_desc = np.empty((dataloader.dataset.__len__(), self.features_dim), dtype=np.float32)
        with torch.no_grad():
            for indicies, batch in tqdm(dataloader, desc=f"Computing {self.name} Query Desc", disable=not pbar):
                features = self.model(batch.to(self.device)).detach().cpu().numpy()
                all_desc[indicies.numpy(), :] = features

        all_desc = all_desc / np.linalg.norm(all_desc, axis=0, keepdims=True)
        query_desc = {"query_descriptors": all_desc}
        self.set_query(query_desc)
        return query_desc

    def compute_map_desc(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader = None,
        pbar: bool = True,
    ) -> dict:
        """
        Compute the map descriptors for the given data.

        Args:
            dataloader (torch.utils.data.dataloader.DataLoader, optional): DataLoader providing the data.
            pbar (bool, optional): If True, display a progress bar. Defaults to True.

        Returns:
            dict: A dictionary containing the computed map descriptors.
        """

        all_desc = np.empty((dataloader.dataset.__len__(), self.features_dim), dtype=np.float32)
        with torch.no_grad():
            for indicies, batch in tqdm(dataloader, desc=f"Computing {self.name} Map Desc", disable=not pbar):
                features = self.model(batch.to(self.device)).detach().cpu().numpy()
                all_desc[indicies.numpy(), :] = features

        all_desc = all_desc / np.linalg.norm(all_desc, axis=0, keepdims=True)
        map_desc = {"map_descriptors": all_desc}
        self.set_map(map_desc)
        return map_desc
