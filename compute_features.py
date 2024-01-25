from train.dataloaders.train.DistillationCitiesDataset import DistillationCitiesDataset
from PlaceRec.utils import get_method
from PlaceRec import utils
import numpy as np
from torch.utils.data import DataLoader 
from os.path import join 
import os
from tqdm import tqdm
from PlaceRec.utils import get_config
import torch
from parsers import train_arguments

config = get_config()
args = train_arguments()


ALL_CITIES = [
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
    "Phoenix",]

DEBUG_CITIES = ["TRT"]


def save_features_batch(features: np.ndarray, names: np.ndarray, dataset_name: str, method_name: str):
    if not os.path.exists(join(config["datasets_directory"], "feature_store", dataset_name, method_name)):
        os.makedirs(join(config["datasets_directory"], "feature_store", dataset_name, method_name))
    for feature, feature_name in zip(features, names):        
        np.save(join(config["datasets_directory"], "feature_store", dataset_name, method_name, feature_name.replace(".jpg", ".npy")), feature)


if __name__ == "__main__":
    ds = DistillationCitiesDataset(cities=ALL_CITIES)
    my_device = "cuda" if torch.cuda.is_available() else "cpu"
    df = ds.dataframe
    img_paths = np.array([ds.base_path + "/Images/" + row["city_id"] + "/" + ds.get_img_name(row) for i, row in df.iterrows()])
    img_names = np.array([img_pth.split('/')[-1] for img_pth in img_paths])


    for method_name in args.teacher_methods: 
        method = get_method(method_name, pretrained=True)
        model = method.model
        model.to(my_device)
        dataset = utils.ImageIdxDataset(img_paths, method.preprocess)
        dataloader = DataLoader(dataset, num_workers=6, pin_memory=True, batch_size=32)
        for batch in tqdm(dataloader, desc="Computing Features"):
            idx, imgs = batch
            with torch.no_grad():
                features = model(imgs.to(my_device)).cpu().numpy()
                names = img_names[idx]
                save_features_batch(features, names, "gsvcities", method.name)


