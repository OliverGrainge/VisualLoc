import pytorch_lightning as pl
from prettytable import PrettyTable
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

from PlaceRec.Training.GSV_Cities.dataloaders.train.GSVCitiesDataset import (
    GSVCitiesDataset, GSVCitiesDistillationDataset)
from PlaceRec.Training.GSV_Cities.dataloaders.val.PittsburghDataset import \
    PittsburghDataset
from PlaceRec.utils import get_config

config = get_config()

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
    def __init__(
        self,
        batch_size=int(config["train"]["batch_size"] / 4),
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False,
        image_size=config["train"]["image_resolution"],
        num_workers=4,
        show_data_stats=True,
        cities=TRAIN_CITIES,
        mean_std=IMAGENET_MEAN_STD,
        batch_sampler=None,
        random_sample_from_each_place=True,
        val_set_names=["pitts30k_val"],
    ):
        super().__init__()
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.shuffle_all = shuffle_all
        self.image_size = image_size
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.show_data_stats = show_data_stats
        self.cities = cities
        self.mean_dataset = mean_std["mean"]
        self.std_dataset = mean_std["std"]
        self.random_sample_from_each_place = random_sample_from_each_place
        self.val_set_names = val_set_names
        self.save_hyperparameters()  # save hyperparameter with Pytorch Lightening

        self.train_transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
            ]
        )

        self.valid_transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
            ]
        )

        self.train_loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "drop_last": False,
            "pin_memory": True,
            "shuffle": self.shuffle_all,
        }

        self.valid_loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers // 2,
            "drop_last": False,
            "pin_memory": True,
            "shuffle": False,
        }

    def setup(self, stage):
        if stage == "fit":
            # load train dataloader with reload routine
            self.reload()

            # load validation sets (pitts_val, msls_val, ...etc)
            self.val_datasets = []
            for valid_set_name in self.val_set_names:
                if "pitts30k_val" in valid_set_name.lower():
                    self.val_datasets.append(
                        PittsburghDataset(
                            which_ds=valid_set_name,
                            input_transform=self.valid_transform,
                        )
                    )
                if "mapillary" in valid_set_name.lower():
                    from PlaceRec.Training.GSV_Cities.dataloaders.val.MapillaryDataset import \
                        MSLS

                    self.val_datasets.append(MSLS(input_transform=self.valid_transform))

                if "sped" in valid_set_name.lower():
                    from PlaceRec.Training.GSV_Cities.dataloaders.val.SpedDataset import \
                        SPEDDataset

                    self.val_datasets.append(
                        SPEDDataset(input_transform=self.valid_transform)
                    )
                if "inria" in valid_set_name.lower():
                    from PlaceRec.Training.GSV_Cities.dataloaders.val.InriaDataset import \
                        InriaDataset

                    self.val_datasets.append(
                        InriaDataset(input_transform=self.valid_transform)
                    )
                if "nordland" in valid_set_name.lower():
                    from PlaceRec.Training.GSV_Cities.dataloaders.val.NordlandDataset import \
                        NordlandDataset

                    self.val_datasets.append(
                        NordlandDataset(input_transform=self.valid_transform)
                    )
                if "essex3in1" in valid_set_name.lower():
                    from PlaceRec.Training.GSV_Cities.dataloaders.val.EssexDataset import \
                        EssexDataset

                    self.val_datasets.append(
                        EssexDataset(input_transform=self.valid_transform)
                    )
                if "corssseasons" in valid_set_name.lower():
                    from PlaceRec.Training.GSV_Cities.dataloaders.val.CrossSeasonDataset import \
                        CrossSeasonDataset

                    self.val_datasets.append(
                        CrossSeasonDataset(input_transform=self.valid_transform)
                    )

            if self.show_data_stats:
                self.print_stats()

    def reload(self):
        self.train_dataset = GSVCitiesDataset(
            cities=self.cities,
            img_per_place=self.img_per_place,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform,
        )

    def train_dataloader(self):
        self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(
                DataLoader(dataset=val_dataset, **self.valid_loader_config)
            )
        return val_dataloaders

    def print_stats(self):
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
        table.add_row(
            ["# of iterations", f"{self.train_dataset.__len__()//self.batch_size}"]
        )
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))


class GSVCitiesDataModuleDistillation(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=int(config["train"]["batch_size"] / 4),
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False,
        teacher_image_size=config["train"]["teacher_resolution"],
        student_image_size=config["train"]["image_resolution"],
        num_workers=4,
        show_data_stats=True,
        cities=TRAIN_CITIES,
        mean_std=IMAGENET_MEAN_STD,
        batch_sampler=None,
        random_sample_from_each_place=True,
        val_set_names=["pitts30k_val"],
    ):
        super().__init__()
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.shuffle_all = shuffle_all
        self.teacher_image_size = teacher_image_size
        self.student_image_size = student_image_size
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.show_data_stats = show_data_stats
        self.cities = cities
        self.mean_dataset = mean_std["mean"]
        self.std_dataset = mean_std["std"]
        self.random_sample_from_each_place = random_sample_from_each_place
        self.val_set_names = val_set_names
        self.save_hyperparameters()  # save hyperparameter with Pytorch Lightening

        self.train_transform = T.Compose(
            [
                T.Resize(
                    self.student_image_size, interpolation=T.InterpolationMode.BILINEAR
                ),
                # T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
            ]
        )

        self.valid_transform = T.Compose(
            [
                T.Resize(
                    self.student_image_size, interpolation=T.InterpolationMode.BILINEAR
                ),
                T.ToTensor(),
                T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
            ]
        )

        self.train_loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "drop_last": False,
            "pin_memory": True,
            "shuffle": self.shuffle_all,
        }

        self.valid_loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers // 2,
            "drop_last": False,
            "pin_memory": True,
            "shuffle": False,
        }

    def setup(self, stage):
        if stage == "fit":
            # load train dataloader with reload routine
            self.reload()

            # load validation sets (pitts_val, msls_val, ...etc)
            self.val_datasets = []
            for valid_set_name in self.val_set_names:
                if "pitts30k" in valid_set_name.lower():
                    self.val_datasets.append(
                        PittsburghDataset(
                            which_ds=valid_set_name,
                            input_transform=self.valid_transform,
                        )
                    )
                if "mapillary" in valid_set_name.lower():
                    from PlaceRec.Training.GSV_Cities.dataloaders.val.MapillaryDataset import \
                        MSLS

                    self.val_datasets.append(MSLS(input_transform=self.valid_transform))

                if "spedtest" in valid_set_name.lower():
                    from PlaceRec.Training.GSV_Cities.dataloaders.val.SpedDataset import \
                        SPEDDataset

                    self.val_datasets.append(
                        SPEDDataset(input_transform=self.valid_transform)
                    )
                if "inria" in valid_set_name.lower():
                    from PlaceRec.Training.GSV_Cities.dataloaders.val.InriaDataset import \
                        InriaDataset

                    self.val_datasets.append(
                        InriaDataset(input_transform=self.valid_transform)
                    )
                if "nordland" in valid_set_name.lower():
                    from PlaceRec.Training.GSV_Cities.dataloaders.val.NordlandDataset import \
                        NordlandDataset

                    self.val_datasets.append(
                        NordlandDataset(input_transform=self.valid_transform)
                    )
                if "essex" in valid_set_name.lower():
                    from PlaceRec.Training.GSV_Cities.dataloaders.val.EssexDataset import \
                        EssexDataset

                    self.val_datasets.append(
                        EssexDataset(input_transform=self.valid_transform)
                    )
                else:
                    print(
                        f"Validation set {valid_set_name} does not exist or has not been implemented yet"
                    )
                    raise NotImplementedError
            if self.show_data_stats:
                self.print_stats()

    def reload(self):
        self.train_dataset = GSVCitiesDistillationDataset(
            cities=self.cities,
            img_per_place=self.img_per_place,
            teacher_img_size=self.teacher_image_size,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform,
        )

    def train_dataloader(self):
        self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(
                DataLoader(dataset=val_dataset, **self.valid_loader_config)
            )
        return val_dataloaders

    def print_stats(self):
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
        table.add_row(
            ["# of iterations", f"{self.train_dataset.__len__()//self.batch_size}"]
        )
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))
