import torch.nn as nn
import torch.nn.init as init
import torch
import numpy as np


from PlaceRec.Methods import RegionVLAD, NetVLAD
from PlaceRec.Datasets import GardensPointWalking

method = RegionVLAD()
ds = GardensPointWalking()


method.load_descriptors(ds.name)

query_images = ds.query_images("test", preprocess=method.preprocess)[:2]

idx, score = method.place_recognise(images=query_images, top_n=2)

print(idx)
print(score)
