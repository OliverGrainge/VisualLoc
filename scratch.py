import torch.nn as nn
import torch.nn.init as init
import torch


from PlaceRec.Methods import AmosNet
from PlaceRec.Datasets import GardensPointWalking

method = AmosNet()
ds = GardensPointWalking()

loader = ds.query_images_loader("train", preprocess=method.preprocess)

q_desc = method.compute_query_desc(dataloader=loader)