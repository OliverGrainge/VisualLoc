from PlaceRec.Methods import DenseVLAD
from PlaceRec.utils import get_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv(
    "/home/oliver/Documents/github/VisualLoc/SelectionData/gsvcities_combinedrecall@1_train.csv"
)


sample_idx = np.random.randint(low=0, high=len(df), size=(10,))


df_sample = df.iloc[sample_idx]

for i in range(10):
    record = df_sample.iloc[i].to_numpy()
    query_img = record[1]
    map_img = record[2]
    print(record[3])
    fig, ax = plt.subplots(2)
    ax[0].imshow(Image.open(query_img))
    ax[1].imshow(Image.open(map_img))
    plt.show()
