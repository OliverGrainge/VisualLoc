from argparse import Namespace
from typing import Dict, List, Tuple, Union

import pytorch_lightning as pl
from prettytable import PrettyTable
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

from train.dataloaders.train.GSVCitiesDataset import (
    GSVCitiesDataset,
)
from train.dataloaders.val.CrossSeasonDataset import (
    CrossSeasonDataset,
)
from train.dataloaders.val.EssexDataset import (
    EssexDataset,
)
from train.dataloaders.val.InriaDataset import (
    InriaDataset,
)
from train.dataloaders.val.MapillaryDataset import MSLS
from train.dataloaders.val.NordlandDataset import (
    NordlandDataset,
)
from train.dataloaders.val.PittsburghDataset import (
    PittsburghDataset,
)
from train.dataloaders.val.SPEDDataset import (
    SPEDDataset,
)

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}

TRAIN_CITIES = [
    "Bangkok",
    "BuenosAires",
    "LosAngeles",
    "MexicoCity",
    "OSL",  # refers to Oslo
    "Rome",
    "Barcelona",
    "Chicago",
    "Madrid",
    "Miami",
    "Phoenix",
    "TRT",  # refers to Toronto
    "Boston",
    "Lisbon",
    "Medellin",
    "Minneapolis",
    "PRG",  # refers to Prague
    "WashingtonDC",
    "Brussels",
    "London",
    "Melbourne",
    "Osaka",
    "PRS",  # refers to Paris
]


class GSVCitiesDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for managing data related to city images from the Google Street View (GSV) dataset.

    Attributes:
        args (Namespace): A configuration namespace containing various parameters.
        batch_size (int): The size of each batch during training.
        img_per_place (int): The number of images per place.
        min_img_per_place (int): The minimum number of images per place.
        shuffle_all (bool): Whether to shuffle the entire dataset.
        image_size (tuple): The size of the images.
        num_workers (int): The number of workers for data loading.
        batch_sampler: Sampler for batching strategies.
        show_data_stats (bool): Flag to indicate if data statistics should be shown.
        cities (list): List of cities included in the dataset.
        mean_dataset (float): The mean value for dataset normalization.
        std_dataset (float): The standard deviation value for dataset normalization.
        random_sample_from_each_place (bool): Flag to sample randomly from each place.
        val_set_names (list): Names of the validation datasets.
        train_loader_config (dict): Configuration dictionary for the training data loader.
        valid_loader_config (dict): Configuration dictionary for the validation data loader.

    Methods:
        setup(stage): Prepares data loaders for the given stage (e.g., 'fit').
        reload(): Reloads the training dataset.
        train_dataloader(): Returns the DataLoader for training.
        val_dataloader(): Returns the DataLoader for validation.
        print_stats(): Prints statistics about the dataset.
    """

    def __init__(self, args: Namespace, mean_std: Dict[str, float] = IMAGENET_MEAN_STD, random_sample_from_each_place: bool = True):
        """
        Initializes the GSVCitiesDataModule with the given arguments.

        Args:
            args (Namespace): A configuration namespace containing various parameters.
            mean_std (dict): A dictionary with 'mean' and 'std' keys for dataset normalization.
            random_sample_from_each_place (bool): If True, samples randomly from each place.
        """
        super().__init__()
        self.args = args
        self.batch_size = args.train_batch_size
        self.img_per_place = args.img_per_place
        self.min_img_per_place = args.min_img_per_place
        self.shuffle_all = args.shuffle_all
        self.image_size = args.image_size
        self.num_workers = args.num_workers
        self.batch_sampler = args.batch_sampler
        self.show_data_stats = args.show_data_stats
        self.cities = TRAIN_CITIES if args.cities[0] == "all" else args.cities
        self.mean_dataset = mean_std["mean"]
        self.std_dataset = mean_std["std"]
        self.random_sample_from_each_place = random_sample_from_each_place
        self.val_set_names = self.args.val_set_names
        self.save_hyperparameters()  # save hyperparameter with Pytorch Lightening

        self.train_transform = T.Compose(
            [
                T.Resize(args.image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
            ]
        )

        self.valid_transform = T.Compose(
            [
                T.Resize(self.args.image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
            ]
        )

        self.train_loader_config = {
            "batch_size": self.args.train_batch_size,
            "num_workers": self.args.num_workers,
            "drop_last": False,
            "pin_memory": True,
            "shuffle": self.args.shuffle_all,
        }

        self.valid_loader_config = {
            "batch_size": self.args.train_batch_size,
            "num_workers": self.args.num_workers // 2,
            "drop_last": False,
            "pin_memory": True,
            "shuffle": False,
        }

    def setup(self, stage: str) -> None:
        """
        Prepares data loaders for the given stage (e.g., 'fit').

        Args:
            stage (str): The stage for which to set up the data module, typically 'fit' or 'test'.
        """
        if stage == "fit":
            # load train dataloader with reload routine
            self.reload()

            # load validation sets (pitts_val, msls_val, ...etc)
            self.val_datasets = []
            for valid_set_name in self.val_set_names:
                if "pitts30k" in valid_set_name.lower():
                    self.val_datasets.append(PittsburghDataset(which_ds=valid_set_name, input_transform=self.valid_transform))
                elif "nordland" in valid_set_name.lower():
                    self.val_datasets.append(NordlandDataset(input_transform=self.valid_transform))
                elif "sped" in valid_set_name.lower():
                    self.val_datasets.append(SPEDDataset(input_transform=self.valid_transform))
                elif "essex" in valid_set_name.lower():
                    self.val_datasets.append(EssexDataset(input_transform=self.valid_transform))
                elif "inria" in valid_set_name.lower():
                    self.val_datasets.append(InriaDataset(input_transform=self.valid_transform))
                elif "cross" in valid_set_name.lower():
                    self.val_datasets.append(CrossSeasonDataset(input_transform=self.valid_transform))
                else:
                    print(f"Validation set {valid_set_name} does not exist or has not been implemented yet")
                    raise NotImplementedError
            if self.show_data_stats:
                self.print_stats()

    def reload(self) -> None:
        """
        Reloads the training dataset. This method is typically called to refresh the dataset, for example, when new data is added.
        """
        self.train_dataset = GSVCitiesDataset(
            cities=self.cities,
            img_per_place=self.img_per_place,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Creates and returns the DataLoader for training.

        Returns:
            DataLoader: The DataLoader for the training dataset.
        """
        self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self) -> List[DataLoader]:
        """
        Creates and returns DataLoaders for validation.

        Returns:
            list[DataLoader]: A list of DataLoaders for each validation dataset.
        """
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(DataLoader(dataset=val_dataset, **self.valid_loader_config))
        return val_dataloaders

    def print_stats(self) -> None:
        """
        Prints statistics about the dataset, including the number of cities, places, images, and details of the training configuration.
        """
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ["Data", "Value"]
        table.align["Data"] = "l"
        table.align["Value"] = "l"
        table.header = False
        table.add_row(["# of cities", f"{len(TRAIN_CITIES)}"])
        table.add_row(["# of places", f"{self.train_dataset.__len__()}"])
        table.add_row(["# of images", f"{self.train_dataset.total_nb_images}"])
        print(table.get_string(title="Training Dataset"))
        print()

        table = PrettyTable()
        table.field_names = ["Data", "Value"]
        table.align["Data"] = "l"
        table.align["Value"] = "l"
        table.header = False
        for i, val_set_name in enumerate(self.val_set_names):
            table.add_row([f"Validation set {i+1}", f"{val_set_name}"])
        # table.add_row(["# of places", f'{self.train_dataset.__len__()}'])
        print(table.get_string(title="Validation Datasets"))
        print()

        table = PrettyTable()
        table.field_names = ["Data", "Value"]
        table.align["Data"] = "l"
        table.align["Value"] = "l"
        table.header = False
        table.add_row(["Batch size (PxK)", f"{self.batch_size}x{self.img_per_place}"])
        table.add_row(["# of iterations", f"{self.train_dataset.__len__()//self.batch_size}"])
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))