import torch
import torch.nn as nn
import torchvision.models as models
import torch_pruning as tp
from PlaceRec import Methods
import torch_pruning as tp
from PlaceRec.utils import get_dataset
import pickle
import pandas as pd
from PlaceRec.utils import get_dataset, get_method
from PlaceRec.Evaluate import Eval

datasets = ["Pitts30k_Val"]
methods = ["MixVPR", "DinoSalad", "EigenPlaces", "SFRS"]

batch_size = 24

columns = ["method_name", "model_memory"]
for ds in datasets:
    columns += [f"{ds}_total_memory", f"{ds}_R1", f"{ds}_map_memory"]
df = pd.DataFrame(columns=columns)

data = []

for method_name in methods:
    method = get_method(method_name, pretrained=True)
    method.set_device("mps")
    row = {}
    for ds_name in datasets:
        ds = get_dataset(ds_name)
        map_loader = ds.map_images_loader(
            preprocess=method.preprocess,
            num_workers=0,
            pin_memory=False,
            batch_size=batch_size,
        )
        _ = method.compute_map_desc(dataloader=map_loader)
        del map_loader
        query_loader = ds.query_images_loader(
            preprocess=method.preprocess,
            num_workers=0,
            pin_memory=False,
            batch_size=batch_size,
        )
        _ = method.compute_query_desc(dataloader=query_loader)
        del query_loader
        method.save_descriptors(ds.name)

        eval = Eval(method, ds)
        eval.compute_all_matches()
        R1 = eval.ratk(1)
        map_memory = eval.map_memory()
        model_memory = eval.model_memory()
        row[f"{ds_name}_total_memory"] = map_memory + model_memory
        row[f"{ds_name}_R1"] = R1
        row[f"{ds_name}_map_memory"] = map_memory

    row["method_name"] = method_name
    row["model_memory"] = model_memory
    data.append(row)


df = pd.DataFrame(data, columns=columns)
df.to_csv("pruning_plots/baselines2.csv")
