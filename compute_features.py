from train.dataloaders.train.GSVCitiesDataset import GSVCitiesDataset
from PlaceRec.utils import get_method
from PlaceRec import utils
import numpy as np
from torch.utils.data import DataLoader 
from os.path import join 
import os

METHODS = ["mixvpr"]


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

if __name__ == "__main__":
    ds = GSVCitiesDataset(cities=TRAIN_CITIES)

    df = ds.dataframe
    print(df.head())
    img_paths = np.array([ds.base_path + "/Images/" + row["city_id"] + "/" + ds.get_img_name(row) for i, row in df.iterrows()])



    for method_name in METHODS: 
        method = get_method(method_name, pretrained=True)
        dataset = utils.ImageIdxDataset(img_paths, method.preprocess)
        dataloader = DataLoader(dataset, num_workers=6, pin_memory=True, batch_size=32)
        features = method.compute_query_desc(dataloader, pbar=True)
        if ds.cities == DEBUG_CITIES:
            features_name = method_name + ".npy"
            np.save(os.getcwd() + "/Data/feature_store/debug/" + features_name, features)
        else:
            features_name = method_name + ".npy"
            np.save(os.getcwd() + "/Data/feature_store/train/" + features_name, features)




