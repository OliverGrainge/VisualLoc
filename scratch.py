import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from torch.utils.data import DataLoader, TensorDataset
import faiss
import numpy as np
import torchvision
from PlaceRec.Training.models import get_backbone, get_aggregator
from typing import Literal
from torchvision import transforms as tvf
import os 
from onedrivedownloader import download
import fast_pytorch_kmeans as fpk
from PIL import Image
import einops as ein
from typing import Union, List, Literal
from torchvision.transforms import functional as T
from PlaceRec.Quantization import quantize_model
from PlaceRec.Training.dataloaders.val.MapillaryDataset import MSLS
from PlaceRec.Training.dataloaders.val.PittsburghDataset import PittsburghDataset
from PlaceRec.Training import valid_transform
from torch.utils.data import Subset
from tqdm import tqdm 
from torch.utils.data import SubsetRandomSampler
from torch.hub import load_state_dict_from_url
from torchvision import transforms
from tqdm import tqdm 




"""
backbone_names = ["resnet101", "resnet50", "resnet18", "efficientnet", "mobilenet", "squeezenet"]
aggregation_names = ["gem", "convap", "cosplace", "mac", "mixvpr", "netvlad", "spoc"]

backbone_names = ["resnet50"]
aggregation_names = ["mac", "spoc"]
img = torch.randn(1, 3, 320, 320)
for back_name in backbone_names:
    backbone = get_backbone(back_name)
    feature_map_shape = backbone(img)[0].shape
    for agg_name in aggregation_names: 
        model = nn.Sequential(get_backbone(back_name), get_aggregator(agg_name, feature_map_shape, out_dim=1024))
        qmodel = quantize_model(backbone, precision="fp32")
        out = qmodel(img)
        print(out.shape)
"""

"""

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super(L2Norm, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Compute the L2 norm (Euclidean norm) of the tensor along the specified dimension
        l2_norm = torch.norm(x, p=2, dim=self.dim, keepdim=True)

        # Normalize the tensor
        x_normalized = x / l2_norm

        return x_normalized
    

    
cal_ds = MSLS(input_transform=valid_transform)
cal_ds = Subset(cal_ds, np.arange(100))
cal_ds_small = Subset(cal_ds, np.arange(10))
cal_dl_small = DataLoader(cal_ds_small, batch_size=2)


img = torch.randn(1, 3, 320, 320).cuda()
#model = torchvision.models.resnet50().eval().cuda()
backbone = get_backbone("resnet50").eval().cuda()
feature_map_shape = backbone(img).detach().cpu()[0].shape
agg = get_aggregator("gem", feature_map_shape=feature_map_shape, out_dim=512)
agg.cuda()

#print("making feature maps")
#feature_maps = torch.vstack([backbone(batch.cuda()) for (batch, idx) in cal_dl_small])
#agg.initialize_netvlad_layer(feature_maps)

model = nn.Sequential(backbone, agg).cuda()
out = model(img)
print(out.shape)
print(out.norm())

model_traced = torch.jit.trace(model, img)
out = model_traced(img)
print(out.shape)
print(out.norm())

qmodel = quantize_model(model_traced, precision="int8", calibration_dataset=cal_ds)
out = qmodel(img)
print(out.shape)
print(out.norm())

"""

from train import VPRModel


#cal_ds = Subset(cal_ds, np.arange(100))


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.dataset1 = MSLS(input_transform=valid_transform)
        self.dataset2 = PittsburghDataset(input_transform=valid_transform)

    def __len__(self):
        # If you want to handle datasets of unequal length, adjust this method
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            # Adjust the index for the second dataset
            return self.dataset2[idx - len(self.dataset1)]

cal_ds = CombinedDataset()

cal_ds_small = Subset(cal_ds, np.arange(100))
cal_dl_small = DataLoader(cal_ds_small, batch_size=2)


#model = VPRModel(backbone_arch='resnet18', agg_arch='netvlad')
#model = VPRModel.load_from_checkpoint(checkpoint_path="/home/oliver/Documents/github/VisualLoc/Checkpoints/resnet18_netvlad.ckpt")


    
img = torch.randn(1, 3, 320, 320).cuda()

#model = VPRModel(backbone_arch='resnet18', agg_arch='netvlad')
#model = VPRModel.load_from_checkpoint(checkpoint_path="/home/oliver/Documents/github/VisualLoc/Checkpoints/resnet18_netvlad.ckpt")
#model = model.model
#model.eval().cuda()

img = torch.randn(1, 3, 320, 320).cuda()
backbone = get_backbone("efficientnet")
backbone.cuda()
feature_map = backbone(img)
#agg = NetVLAD(feature_map_shape=feature_map[0].shape)
agg = get_aggregator("netvlad", feature_map[0].shape)
agg.eval().cuda()

model = nn.Sequential(backbone, agg)
model.eval().cuda()

out = model(img)
print(out.shape)
print(out.norm())

model_traced = torch.jit.trace(model, img)
out = model_traced(img)
print(out.shape)
print(out.norm())

batch = cal_ds_small.__getitem__(0)
print(batch[0].shape, batch[1].shape)

print("================================", cal_ds.__len__())

import time 
st = time.time()

qmodel = quantize_model(model, precision="int8", calibration_dataset=cal_ds_small)
img = torch.randn(1, 3, 320, 320).cuda().float()
print(img.shape)
out = qmodel(img)
print(out.shape)
print(out.norm())

ed = time.time()

print("Quantizing time", ed - st)











