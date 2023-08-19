import torch.nn as nn
import torch.nn.init as init
import torch
import numpy as np


from PlaceRec.Methods import CALC
from PlaceRec.Datasets import GardensPointWalking

method = CALC()
ds = GardensPointWalking()

loader = ds.query_images_loader("train", preprocess=method.preprocess)


q_desc = method.compute_query_desc(dataloader=loader)
