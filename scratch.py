from PlaceRec.Methods import DenseVLAD
from PlaceRec.utils import get_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from itertools import combinations

df = pd.read_csv(
    "/home/oliver/Documents/github/VisualLoc/SelectionData/gsvcities_combinedrecall@1_test.csv"
)

"""

methods = ["netvlad", "hybridnet", "amosnet"]

comb = combinations(methods, 2)


for c in comb:
    method1 = c[0]
    method2 = c[1]

    scores1 = df[method1].to_numpy().astype(np.float32)
    scores2 = df[method2].to_numpy().astype(np.float32)

    print(
        method1,
        np.sum(scores1) / len(scores1),
        method2,
        np.sum(scores2) / len(scores2),
        " ============",
        method1 + " + " + method2,
        np.sum(np.logical_or(scores1, scores2) / len(scores1)),
    )

"""
netvlad_correct = df[(df["netvlad"] == 1.0) & (df["hybridnet"] == 0.0)]
amosnet_correct = df[(df["hybridnet"] == 1.0) & (df["netvlad"] == 0.0)]

netvlad_correct = netvlad_correct.sample(frac=1).reset_index(drop=True)
amosnet_correct = amosnet_correct.sample(frac=1).reset_index(drop=True)

max_idx = min(len(netvlad_correct), len(amosnet_correct))
netvlad_sample = netvlad_correct.head(max_idx)
amosnet_sample = amosnet_correct.head(max_idx)

df = pd.concat((netvlad_sample, amosnet_sample))

df = df.sample(frac=1).reset_index(drop=True)
df.to_csv(
    "/home/oliver/Documents/github/VisualLoc/SelectionData/gsvcities_combinedrecall@1_oneright_test.csv"
)


df = pd.read_csv(
    "/home/oliver/Documents/github/VisualLoc/SelectionData/gsvcities_combinedrecall@1_oneright_test.csv"
)

netvlad_count = np.sum(df["netvlad"].to_numpy())
amosnet_count = np.sum(df["hybridnet"].to_numpy())


print("netvlad positives", netvlad_count, "amosnet positives", amosnet_count)
