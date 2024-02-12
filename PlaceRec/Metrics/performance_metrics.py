import os
import sys
import time
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from thop import profile
from PIL import Image
import pytorch_lightning as pl

from PlaceRec.Datasets import GardensPointWalking




def measure_memory(args, method): 
    engine_path = "tensorrt_engine.trt"
    try:
        # Get the size of the engine file in bytes
        file_size_bytes = os.path.getsize(engine_path)
        return file_size_bytes / (1024**2)
    except Exception as e:
        print(f"Error obtaining file size: {e}")
        return None



def benchmark_latency_cpu(method, num_runs=100):
    """
    Benchmark the inference latency of a PyTorch model on the CPU.

    Parameters:
    - method: VPR technique to benchmark.
    - num_runs: Number of runs to average over.

    Returns:
    - Average inference time in milliseconds.
    """
    model = method.model
    model.eval()
    model = model.cpu()
   
    img = np.random.randint(0, 255, size=(480, 640, 3)).astype(np.uint8)
    img = Image.fromarray(img)
    img = method.preprocess(img)
    img = img[None, :]

    for _ in range(10):
        _ = model(img)

    # Measure inference time
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(img)
    end_time = time.time()
    average_time = (end_time - start_time) / num_runs * 1000  # Convert to milliseconds
    return average_time


def benchmark_latency_gpu(method, num_runs: int = 100):
    """
    Benchmark the inference latency of a PyTorch model using CUDA events.

    Parameters:
    - method: VPR Method to benchmark..
    - num_runs: Number of runs to average over.

    Returns:
    - Average inference time in milliseconds.
    """
    # Ensure the model and input data are on the GPU
    img = np.random.randint(0, 255, size=(480, 640, 3)).astype(np.uint8)
    img = Image.fromarray(img)
    img = method.preprocess(img)
    img = img[None, :]

    model = method.model
    img = img.cuda()
    # Warm up
    for _ in range(20):
        _ = model(img)
    
    # Measure inference time using CUDA events
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()  # Ensure CUDA operations are synchronized
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        _ = model(img)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    return np.mean(times)


def count_params(method) -> int:
    """
    Counts the total number of parameters for the model in a given method.

    Parameters:
    - method: Method object which contains a model.

    Returns:
    - int: Total number of parameters.
    """

    assert isinstance(method.model, nn.Module)
    if method.model is not None:
        total_params = sum(p.numel() for p in method.model.parameters())
        return int(total_params)
    else:
        return 0


def count_flops(method) -> int:
    """
    Counts the number of floating point operations (FLOPs) for a given method.

    Parameters:
    - method: Method object which contains a model and preprocessing function.

    Returns:
    - int: Number of FLOPs.
    """

    from PlaceRec.Datasets import GardensPointWalking

    method.model.eval()
    img = np.random.randint(0, 255, size=(480, 640, 3)).astype(np.uint8)
    img = Image.fromarray(img)
    img = method.preprocess(img)
    img = img[None, :]
    assert isinstance(method.model, nn.Module)
    input = img.to(method.device)  # get one input item
    flops, _ = profile(method.model, inputs=(input,), verbose=False)
    return int(flops)



