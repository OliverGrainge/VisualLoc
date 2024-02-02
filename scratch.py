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
import time
import statistics
import tensorrt as trt
from onnxconverter_common import auto_convert_mixed_precision
from train import VPRModel


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
""""""
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType, CalibrationDataReader
import onnxruntime as ort
import numpy as np 
from onnxconverter_common import float16
from onnxruntime import quantization
from PlaceRec.Quantization import quant


def get_model(backbone_arch, aggregation_arch, out_dim=1024):
    img = torch.randn(1, 3, 320, 320)
    backbone = get_backbone(backbone_arch)
    feature_map = backbone(img)
    agg = get_aggregator(aggregation_arch, feature_map[0].shape, out_dim=out_dim)
    model = nn.Sequential(backbone, agg)
    return model


def time_execution(qmodel, img, warmup_runs=3, timed_runs=10):
    """
    Times the execution of qmodel(img) with warmup, using high-resolution timer
    and statistical analysis.

    :param qmodel: The model function to be timed.
    :param img: The image input for the model.
    :param warmup_runs: Number of warmup runs before timing.
    :param timed_runs: Number of runs to time.
    :return: Average execution time and standard deviation.
    """
    # Warmup phase
    for _ in range(warmup_runs):
        _ = qmodel(img)

    # Timing phase
    times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(timed_runs):
        start_event.record()
        _ = qmodel(img)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    # Statistical analysis
    average_time = statistics.mean(times)
    return average_time


img = torch.randn(10, 3, 320, 320)
cal_ds = Subset(MSLS(valid_transform), list(range(100)))
model = get_model("efficientnet", "spoc", 1024)
#method = VPRModel.load_from_checkpoint("/home/oliver/Documents/github/VisualLoc/Checkpoints/resnet18_netvlad_1024.ckpt")
#model = method.model
assert isinstance(model, nn.Module)



avg = time_execution(model.cuda(), img.cuda(), timed_runs=1000)
print("Full precision pt", avg)
avg = time_execution(model.cuda().half(), img.cuda().half(), timed_runs=1000)
print("Half precision pt", avg)

qmodel_int8 = quantize_model(model, precision="int8", calibration_dataset=cal_ds, batch_size=10)
avg_int8 = time_execution(qmodel_int8, img.float().cuda(), timed_runs=1000)

qmodel_fp16 = quantize_model(model, precision="fp16", calibration_dataset=cal_ds, batch_size=10)
avg_fp16= time_execution(qmodel_fp16, img.float().cuda(), timed_runs=1000)

qmodel_fp32 = quantize_model(model, precision="fp32", calibration_dataset=cal_ds, batch_size=10)
avg_fp32= time_execution(qmodel_fp32, img.float().cuda(), timed_runs=1000)

print("tensorrt fp32", avg_fp32)
print("tensorrt fp16", avg_fp16)
print("tensorrt int8", avg_int8)

