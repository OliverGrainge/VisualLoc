import time

import torch
import torch_pruning as tp

from PlaceRec.utils import get_method
import torch
from torch.nn.utils import prune
import torch.nn as nn

from apex.contrib.sparsity import ASP

method = get_method("resnet50_eigenplaces", pretrained=True)

model = method.model
model.eval()
model.to("cpu")

optimizer = torch.nn.optim.Adam(model.parameters(), lr=0.001)

ASP.prune_trained_model(model, optimizer)

