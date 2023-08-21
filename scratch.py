import torch.nn as nn
import torch.nn.init as init
import torch
import numpy as np


from PlaceRec.Methods import RegionVLAD, NetVLAD
from PlaceRec.Datasets import GsvCities

ds = GsvCities()


gt = ds.ground_truth("test", gt_type="hard")
gts = ds.ground_truth("test", gt_type="soft")

qp = ds.query_partition("test")
mp = ds.map_partition("test")

print(len(mp), len(qp))

print(gt.shape, gts.shape)
