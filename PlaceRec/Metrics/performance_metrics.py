import os
import sys
import time
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from thop import profile

from PlaceRec.Datasets import GardensPointWalking


def measure_memory(args, method, jit=True):
    """
    Measures the size (in bytes) of a given PyTorch model when saved to disk.

    Parameters:
    - args (object): Placeholder for any additional arguments. Currently not used in the function.
    - model (torch.nn.Module or torch.jit.ScriptModule): The PyTorch model to measure. This can be
        a regular PyTorch model or a JIT scripted model.
    - jit (bool, optional): Indicates whether the given model is a JIT scripted model.
        If True, saves the model using torch.jit.save. If False, saves the model's state_dict.
        Default is True.

    Returns:
    - int: Size of the saved model in mega bytes.

    Note:
    - This function temporarily saves the model to 'tmp_model.pt' on disk to measure its size.
        The temporary file is deleted after size measurement.
    """
    model = method.model.to(args.device)
    model.eval()
    if jit:
        if isinstance(model, nn.Module):
            example_input = torch.randn(args.input_size).numpy().astype(np.float32)
            example_input = method.preprocess(example_input).to(args.device)
            traced_model = torch.jit.trace(model, example_input[None,:])
            model = torch.jit.script(traced_model)
        torch.jit.save(model, "tmp_model.pt")
        size_in_bytes = os.path.getsize("tmp_model.pt")
        os.remove("tmp_model.pt")
        return size_in_bytes / 1000000
    else:
        torch.save(model.state_dict(), "tmp_model.pt")
        size_in_bytes = os.path.getsize("tmp_model.pt")
        os.remove("tmp_model.pt")
        return size_in_bytes / 1000000


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
    ds = GardensPointWalking()
    dl = ds.query_images_loader("test", preprocess=method.preprocess)
    for batch in dl:
        break
    input_data = batch[0].unsqueeze(0)
    for _ in range(10):
        _ = model(input_data)

    # Measure inference time
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(input_data)
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
    ds = GardensPointWalking()
    dl = ds.query_images_loader("test", preprocess=method.preprocess)
    for batch in dl:
        break
    input_data = batch[0].unsqueeze(0)
    model = model.cuda()
    input_data = input_data.cuda()
    model.eval()
    # Warm up
    for _ in range(10):
        _ = model(input_data)
    torch.cuda.synchronize()  # Ensure CUDA operations are synchronized
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # Measure inference time using CUDA events
    start_event.record()
    for _ in range(num_runs):
        _ = model(input_data)
    end_event.record()
    torch.cuda.synchronize()
    average_time = start_event.elapsed_time(end_event) / num_runs
    return average_time


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
    ds = GardensPointWalking()
    loader = ds.query_images_loader("test", preprocess=method.preprocess)
    assert isinstance(method.model, nn.Module)
    if method.model is not None:
        for batch in loader:
            input = batch[0][None, :].to(method.device)  # get one input item
            flops, _ = profile(method.model, inputs=(input,))
            return int(flops)
    else:
        return 0
