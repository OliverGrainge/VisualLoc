import os
import zipfile
from glob import glob

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from scipy.signal import convolve2d
from torch.utils.data import DataLoader

from PlaceRec.utils import ImageIdxDataset, s3_bucket_download
from PlaceRec.Datasets.base_dataset import BaseDataset

package_directory = os.path.dirname(os.path.abspath(__file__))


class GsvBase:
    def __init__(
        self,
        root="/home/oliver/Documents/github/Datasets/gsv_cities/",
        cities: list = ["London"],
        min_img_per_place: int = 4,
    ):
        self.root = root
        self.cities = cities
        self.min_img_per_place = min_img_per_place
        self.dataframe = self.get_fulldataframe()
        self.places_ids = pd.unique(self.dataframe.index)
        self.n_places = len(self.places_ids)
        self.total_nb_images = len(self.dataframe)

    def get_place_dataframe(self, idx):
        place_id = self.places_ids[idx]
        return self.dataframe.loc[place_id]

    def split_place(self, df):
        df = df.head(4)
        # boundary = int(len(df.index) * 0.5)
        # query, reference = df[:boundary], df[boundary:]
        query, reference = df.iloc[:2], df.iloc[2:]
        return query, reference

    def get_fulldataframe(self):
        """
        Return one dataframe containing
        all info about the images from all cities

        This requieres DataFrame files to be in a folder
        named Dataframes, containing a DataFrame
        for each city in self.cities
        """
        # read the first city dataframe
        df = pd.read_csv(self.root + "Dataframes/" + f"{self.cities[0]}.csv")

        # append other cities one by one
        for i in range(1, len(self.cities)):
            tmp_df = pd.read_csv(self.root + "Dataframes/" + f"{self.cities[i]}.csv")

            # Now we add a prefix to place_id, so that we
            # don't confuse, say, place number 13 of NewYork
            # with place number 13 of London ==> (0000013 and 0500013)
            # We suppose that there is no city with more than
            # 99999 images and there won't be more than 99 cities
            # TODO: rename the dataset and hardcode these prefixes
            prefix = i
            tmp_df["place_id"] = tmp_df["place_id"] + (prefix * 10**5)
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe

            df = pd.concat([df, tmp_df], ignore_index=True)

        # keep only places depicted by at least min_img_per_place images
        res = df[df.groupby("place_id")["place_id"].transform("size") >= self.min_img_per_place]
        return res.set_index("place_id")

    def get_img_name(self, row):
        # given a row from the dataframe
        # return the corresponding image name

        city = row["city_id"]

        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        pl_id = row.name % 10**5  # row.name is the index of the row, not to be confused with image name
        pl_id = str(pl_id).zfill(7)

        panoid = row["panoid"]
        year = str(row["year"]).zfill(4)
        month = str(row["month"]).zfill(2)
        northdeg = str(row["northdeg"]).zfill(3)
        lat, lon = str(row["lat"]), str(row["lon"])
        name = city + "_" + pl_id + "_" + year + "_" + month + "_" + northdeg + "_" + lat + "_" + lon + "_" + panoid + ".jpg"
        return self.root + "Images/" + city + "/" + name

    def get_paths(self, df):
        names = [self.get_img_name(df.iloc[i]) for i in range(len(df.index))]
        return names

    def load_data(self, cities):
        query_place_id = []
        map_place_id = []
        query_paths = []
        map_paths = []
        for idx in range(self.n_places):
            place = self.get_place_dataframe(idx)
            query_place, map_place = self.split_place(place)
            query_names = self.get_paths(query_place)
            map_names = self.get_paths(map_place)
            a = [idx for _ in range(len(query_names))]
            b = [idx for _ in range(len(map_names))]
            query_place_id += [idx for _ in range(len(query_names))]
            map_place_id += [idx for _ in range(len(map_names))]
            query_paths += query_names
            map_paths += map_names
        return query_paths, map_paths, query_place_id, map_place_id


class GsvCities(BaseDataset, GsvBase):
    def __init__(
        self,
        root: str = "/home/oliver/Documents/github/Datasets/gsv_cities/",
        cities: list = ["London", "Boston"],
        min_img_per_place: int = 4,
    ):
        super().__init__(root=root, cities=cities, min_img_per_place=min_img_per_place)
        if not os.path.isdir(package_directory + "/raw_images/ESSEX3IN1"):
            raise Exception("Dataset not downloaded")

        qp, mp, qid, mid = self.load_data(self.cities)
        self.query_paths = qp
        self.map_paths = mp
        self.query_place_ids = qid
        self.map_place_ids = mid

        q_size = len(self.query_place_ids)
        self.train_ids = qid[: int(q_size * 0.8)]
        self.val_ids = qid[int(q_size * 0.8) : int(q_size * 0.9)]
        self.test_ids = qid[int(q_size * 0.9) :]

        self.name = "gsvcities"

    def query_partition(self, partition: str) -> np.ndarray:
        size = len(self.query_paths)

        # get the required partition of the dataset
        if partition == "train":
            paths = self.query_paths[: int(size * 0.6)]
        elif partition == "val":
            paths = self.query_paths[int(size * 0.8) : int(size * 0.9)]
        elif partition == "test":
            paths = self.query_paths[int(size * 0.9) :]
        elif partition == "all":
            paths = self.query_paths
        else:
            raise Exception("Partition must be 'train', 'val' or 'all'")

        return np.array(paths)

    def map_partition(self, partition: str) -> np.ndarray:
        return np.array(self.map_paths)

    def query_images(
        self,
        partition: str,
        preprocess: torchvision.transforms.transforms.Compose = None,
    ) -> torch.Tensor:
        paths = self.query_partition(partition)

        if preprocess == None:
            return np.array([np.array(Image.open(pth).resize((720, 720))) for pth in paths])
        else:
            imgs = np.array([np.array(Image.open(pth).resize((720, 720))) for pth in paths])
            return torch.stack([preprocess(q) for q in imgs])

    def map_images(
        self,
        partition: str,
        preprocess: torchvision.transforms.transforms.Compose = None,
    ) -> torch.Tensor:
        if preprocess == None:
            return np.array([np.array(Image.open(pth).resize((720, 720))) for pth in self.map_paths])
        else:
            imgs = np.array([np.array(Image.open(pth).resize((720, 720))) for pth in self.map_paths])
            return torch.stack([preprocess(q) for q in imgs])

    def query_images_loader(
        self,
        partition: str,
        batch_size: int = 16,
        shuffle: bool = False,
        preprocess: torchvision.transforms.transforms.Compose = None,
        pin_memory: bool = False,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        paths = self.query_partition(partition)

        # build the dataloader
        dataset = ImageIdxDataset(paths, preprocess=preprocess)
        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        return dataloader

    def map_images_loader(
        self,
        partition: str,
        batch_size: int = 16,
        shuffle: bool = False,
        preprocess: torchvision.transforms.transforms.Compose = None,
        pin_memory: bool = False,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        paths = self.map_partition(partition)

        # build the dataloader
        dataset = ImageIdxDataset(paths, preprocess=preprocess)
        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        return dataloader

    def ground_truth(self, partition: str, gt_type: str) -> np.ndarray:
        if partition == "train":
            q_ids = self.train_ids
        elif partition == "test":
            q_ids = self.test_ids
        elif partition == "val":
            q_ids = self.val_ids
        else:
            q_ids = self.query_place_ids

        m_ids = self.map_place_ids
        ground_truth = np.array(m_ids)[:, np.newaxis] == np.array(q_ids)
        return ground_truth
