import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from PlaceRec.Deploy import (
    deploy_cpu,
    deploy_gpu,
    deploy_tensorrt,
    deploy_tensorrt_sparse,
)
from PlaceRec.Methods import CCT_CLS, ConvAP, MixVPR, ViT_CLS


def measure_latency_gpu(method, batch_size=1, num_runs=100):
    """
    Measure the latency of a model's forward pass on GPU, in milliseconds.

    Args:
    - model: The PyTorch model to measure.
    - input_tensor: A tensor containing input to the model. Make sure it's on the same device as the model.
    - num_runs: The number of runs to average over.

    Returns:
    - Average latency in milliseconds.
    """
    img = method.example_input().cuda().repeat(batch_size, 1, 1, 1)
    if isinstance(method.model, nn.Module):
        method.model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = method.inference(img)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    elapsed_time = 0.0
    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc=f"GPU Latency {method.name}"):
            start_event.record()
            _ = method.inference(img)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time += start_event.elapsed_time(end_event)
    avg_time = elapsed_time / num_runs
    return avg_time


@torch.no_grad()
def measure_latency_cpu(method, num_runs=100):
    img = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
    img = Image.fromarray(img)
    img = method.preprocess(img)
    img = img.to("cpu")
    if isinstance(method.model, nn.Module):
        method.set_device("cpu")
        method.model.eval()

    for _ in range(10):
        method.inference(img[None, :])
    times = []
    for _ in tqdm(range(num_runs), desc=f"CPU Latency {method.name}"):
        st = time.perf_counter()
        method.inference(img[None, :])
        et = time.perf_counter()
        times.append((et - st) * 1000)
    return np.mean(times)


"""
method = ViT_CLS()

method = deploy_cpu(method, sparse=False)
lat1 = measure_latency_cpu(method)

method.load_weights(
    "/media/oliver/9e8e8649-c1f2-4524-9025-f2c751d67f57/home/oliver/Documents/Checkpoints/gsv_cities_sparse_unstructured/vit_cls/vit_cls_epoch[209]_step[131460]_R1[0.8649]_sparsity[96.10].ckpt"
)
method = deploy_cpu(method, sparse=True)
lat2 = measure_latency_gpu(method)

print(f"Dense Latency {lat1} Sparse Latency {lat2}")
"""


method = MixVPR()

method = deploy_gpu(method, batch_size=1, sparse=False)
lat1 = measure_latency_gpu(method, batch_size=1)

method.load_weights(
    "/media/oliver/9e8e8649-c1f2-4524-9025-f2c751d67f57/home/oliver/Documents/Checkpoints/gsv_cities_sparse_semistructured/mixvpr/mixvpr_epoch[14]_step[9390]_R1[0.9171]_SPARSITY[0.513].ckpt"
)

for param in method.model.parameters():
    print(param.data)

method = deploy_gpu(method, batch_size=1, sparse=True)
lat2 = measure_latency_gpu(method, batch_size=1)

print(f"Dense Latency {lat1} Sparse Latency {lat2}")
