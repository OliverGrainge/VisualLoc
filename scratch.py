import torch
import pandas as pd
from PlaceRec.Evaluate import Eval
from os.path import join
from tqdm import tqdm

directory = "/home/oliver/Downloads/ResNet34_Checkpoints"

df = pd.read_csv("pruning_plots/results.csv")
df.set_index("weight_path", inplace=True)


img = torch.randn(1, 3, 320, 320)

for index, row in tqdm(df.iterrows()):
    path = join(directory, index)
    model = torch.load(path, map_location="cpu")
    out = model(img)
    df.loc[index, "descriptor_dim"] = out.shape[1]
    print(index, out.shape[1])


df_filt = df[df["agg_rate"] == 0.0]

print(df_filt.head())

print(df_filt["descriptor_dim"])
