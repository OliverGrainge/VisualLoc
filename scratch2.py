import pandas as pd 
import numpy as np


df = pd.read_csv("Plots/PlaceRec/data/results.csv")
df.head()
df.set_index("id", inplace=True)

precisions = ["int8", "fp16", "fp32"]
backbones = ["resnet18", "resnet50", "resnet101", "vgg16", "mobilenet", "squeezenet", "efficientnet", "dinov2"]
aggregations = ["spoc", "mac", "gem", "mixvpr", "netvlad"]


for back in backbones:
    for agg in aggregations:
        for prec in precisions:
            key = f'{back}_{agg}_1024_{prec}'
            row = df.loc[key]
            if np.isnan(row["stlucia_recall@1"]):
                print(key)