from os.path import join
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from PlaceRec.utils import get_config

config = get_config()

default_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# NOTE: Hard coded path to dataset folder
BASE_PATH = join(config["datasets_directory"], "gsv_cities")

if not Path(BASE_PATH).exists():
    raise FileNotFoundError("BASE_PATH is hardcoded, please adjust to point to gsv_cities")


class GSVCitiesDataset(Dataset):
    """
    A dataset class for loading and processing images from various cities.

    This class extends PyTorch's Dataset class to handle datasets consisting of images
    from different cities. It supports random sampling of images from each city and
    allows for a minimum number of images per place to be specified.

    Attributes:
        base_path (str): The base path to the dataset directory.
        cities (List[str]): A list of city names to include in the dataset.
        img_per_place (int): The number of images to sample from each place.
        min_img_per_place (int): The minimum number of images required for a place to be included.
        random_sample_from_each_place (bool): Flag to randomize image sampling from each place.
        transform (Compose): A torchvision Compose object for image transformations.
        dataframe (pd.DataFrame): The dataframe containing image metadata.
        places_ids (pd.Index): Index of unique place IDs in the dataset.
        total_nb_images (int): Total number of images in the dataset.

    Methods:
        __init__: Initializes the dataset with given parameters.
        __getdataframes: Creates a dataframe with image metadata from all cities.
        __getitem__: Returns a batch of images and their associated place IDs.
        __len__: Returns the total number of unique places in the dataset.
        image_loader: Loads an image from a given path.
        get_img_name: Generates an image file name based on metadata.

    Parameters:
        cities (List[str]): List of city names.
        img_per_place (int): Number of images per place.
        min_img_per_place (int): Minimum number of images for a place to be included.
        random_sample_from_each_place (bool): Whether to randomly sample images.
        transform (Compose): Image transformation operations.
        base_path (str): Base directory path for the dataset.
    """

    def __init__(
        self,
        cities: List[str] = ["London", "Boston"],
        img_per_place: int = 4,
        min_img_per_place: int = 4,
        random_sample_from_each_place: bool = True,
        transform: Compose = default_transform,
        base_path: str = BASE_PATH,
    ) -> None:
        super(GSVCitiesDataset, self).__init__()
        self.base_path = base_path
        self.cities = cities
        assert img_per_place <= min_img_per_place, f"img_per_place should be less than {min_img_per_place}"
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        self.dataframe = self.__getdataframes()
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)

    def __getdataframes(self) -> pd.DataFrame:
        """Creates and returns a consolidated dataframe of image metadata from all specified cities."""
        df = pd.read_csv(self.base_path + "/Dataframes/" + f"{self.cities[0]}.csv")
        df = df.sample(frac=1)  # shuffle the city dataframe
        for i in range(1, len(self.cities)):
            tmp_df = pd.read_csv(self.base_path + "/Dataframes/" + f"{self.cities[i]}.csv")
            prefix = i
            tmp_df["place_id"] = tmp_df["place_id"] + (prefix * 10**5)
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe
            df = pd.concat([df, tmp_df], ignore_index=True)
        res = df[df.groupby("place_id")["place_id"].transform("size") >= self.min_img_per_place]
        return res.set_index("place_id")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a batch of images and their associated place IDs based on the given index.

        Parameters:
            index (int): The index of the place to retrieve images from.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A batch of images and their corresponding place IDs.
        """
        place_id = self.places_ids[index]
        place = self.dataframe.loc[place_id]
        if self.random_sample_from_each_place:
            place = place.sample(n=self.img_per_place)
        else:  # always get the same most recent images
            place = place.sort_values(by=["year", "month", "lat"], ascending=False)
            place = place[: self.img_per_place]
        imgs = []
        for i, row in place.iterrows():
            img_name = self.get_img_name(row)
            img_path = self.base_path + "/Images/" + row["city_id"] + "/" + img_name
            img = self.image_loader(img_path)
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place)

    def __len__(self) -> int:
        """Returns the total number of unique places in the dataset."""
        return len(self.places_ids)

    @staticmethod
    def image_loader(path: str) -> Image:
        """
        Loads an image from the specified path.

        Parameters:
            path (str): The file path of the image to load.

        Returns:
            Image: The loaded image.
        """
        return Image.open(path).convert("RGB")

    @staticmethod
    def get_img_name(row: pd.Series) -> str:
        """
        Generates a unique image file name based on the metadata in the given row.

        Parameters:
            row (pd.Series): A series containing metadata of an image.

        Returns:
            str: The generated image file name.
        """
        city = row["city_id"]
        pl_id = row.name % 10**5  # row.name is the index of the row, not to be confused with image name
        pl_id = str(pl_id).zfill(7)
        panoid = row["panoid"]
        year = str(row["year"]).zfill(4)
        month = str(row["month"]).zfill(2)
        northdeg = str(row["northdeg"]).zfill(3)
        lat, lon = str(row["lat"]), str(row["lon"])
        name = city + "_" + pl_id + "_" + year + "_" + month + "_" + northdeg + "_" + lat + "_" + lon + "_" + panoid + ".jpg"
        return name
