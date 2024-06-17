import torch
import torch.nn as nn
import torchvision.models as models
import torch_pruning as tp
from PlaceRec import Methods


aggregations = ["0.0", "0.5", ""]

weights = "/Users/olivergrainge/Downloads/Checkpoints/ResNet50_GeM_agg_2.00_sparsity_0.657_R1_0.786.ckpt"

mod = torch.load(weights, map_location="cpu")
print(mod)
