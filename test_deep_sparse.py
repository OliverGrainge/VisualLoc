from PlaceRec.Methods import MixVPR, CCT_CLS, ViT_CLS, ConvAP
from PlaceRec.Deploy import deploy_tensorrt
from PlaceRec.Deploy import deploy_tensorrt_sparse
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch.nn as nn


def measure_latency_gpu(method, num_runs=100):
    """
    Measure the latency of a model's forward pass on GPU, in milliseconds.

    Args:
    - model: The PyTorch model to measure.
    - input_tensor: A tensor containing input to the model. Make sure it's on the same device as the model.
    - num_runs: The number of runs to average over.

    Returns:
    - Average latency in milliseconds.
    """
    img = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
    img = Image.fromarray(img)
    img = method.preprocess(img)
    img = img.to("cuda")
    if isinstance(method.model, nn.Module):
        method.model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = method.inference(img[None, :])
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    elapsed_time = 0.0
    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc=f"GPU Latency {method.name}"):
            start_event.record()
            _ = method.inference(img[None, :])
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time += start_event.elapsed_time(end_event)
    avg_time = elapsed_time / num_runs
    return avg_time


method = ViT_CLS()

method = deploy_tensorrt(method)
lat1 = measure_latency_gpu(method)

method.load_weights(
    "/media/oliver/9e8e8649-c1f2-4524-9025-f2c751d67f57/home/oliver/Documents/Checkpoints/gsv_cities_sparse_unstructured/vit_cls/vit_cls_epoch[209]_step[131460]_R1[0.8649]_sparsity[96.10].ckpt"
)
method = deploy_tensorrt_sparse(method)
lat2 = measure_latency_gpu(method)

print(f"Dense Latency {lat1} Sparse Latency {lat2}")
