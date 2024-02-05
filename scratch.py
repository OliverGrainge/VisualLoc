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
from PlaceRec.Quantization import quantize_model_trt


from PlaceRec.Training.models import get_model

model = get_model("vgg16", "gem", descriptor_size=1024)
qmodel = quantize_model_trt(model, precision="int8", force_recalibration=True, descriptor_size=1024)
img = torch.randn(1, 3, 320, 320).cuda()
out = qmodel(img)
print(out.shape)