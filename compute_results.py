# from PlaceRec import Methods, Datasets
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from os.path import join

import numpy as np
import psutil
import torch
import torch.nn as nn
from PIL import Image
from torchprofile import profile_macs
from tqdm import tqdm

from PlaceRec.Deploy import deploy_onnx_cpu, deploy_tensorrt_sparse
from PlaceRec.Evaluate import Eval
from PlaceRec.utils import get_dataset, get_method

CheckpointDirectory = "/media/oliver/9e8e8649-c1f2-4524-9025-f2c751d67f57/home/oliver/Documents/Checkpoints"
METHODS = ["vit_cls", "cct_cls", "mixvpr", "convap"]
DATASETS = ["pitts30k", "spedtest", "gardenspointwalking", "sfu"]

sparsity_type = {
    "unstructured": "gsv_cities_sparse_unstructured/",
    # "semistructured": "gsv_cities_sparse_semistructured/",
    # "structured": "gsv_cities_sparse_structured/",
}


def extract_sparsity(path_name):
    pattern = r"sparsity\[([\d\.]+)\]"
    match = re.search(pattern, path_name)
    if match:
        return float(match.group(1))
    else:
        pattern = r"SPARSITY\[([\d\.]+)\]"
        match = re.search(pattern, path_name)
        if match:
            return float(match.group(1))
        else:
            return None


def extract_density(path_name):
    pattern = r"sparsity\[([\d\.]+)\]"
    match = re.search(pattern, path_name)
    if match:
        return (1 - float(match.group(1))) * 100
    else:
        pattern = r"SPARSITY\[([\d\.]+)\]"
        match = re.search(pattern, path_name)
        if match:
            return (1 - float(match.group(1))) * 100
        else:
            return None


def extract_recall(path_name):
    pattern = r"_R1\[([\d\.]+)\]"
    match = re.search(pattern, path_name)
    if match:
        return float(match.group(1))
    else:
        return None


@torch.no_grad()
def measure_latency_cpu(method, num_runs=1000):
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


def measure_latency_gpu(method, num_runs=2000):
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


def measure_flops(method):
    dev = method.device
    method.set_device("cpu")
    img = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
    img = Image.fromarray(img)
    img = method.preprocess(img)

    flops = profile_macs(method.model, (img[None, :],))
    method.set_device(dev)
    return flops


results = {}
for st in list(sparsity_type.keys()):
    directory = join(CheckpointDirectory, sparsity_type[st])
    methods_list = os.listdir(directory)
    results[st] = {}
    for method_name in methods_list:
        if method_name in METHODS:
            results[st][method_name] = {}
            method = get_method(method_name, pretrained=False)
            weight_dir = join(directory, method_name + "/")
            weight_names = os.listdir(weight_dir)

            for dataset in DATASETS:
                results[st][method_name][dataset] = defaultdict(list)

                for weight in weight_names:
                    weight_path = join(weight_dir, weight)
                    # Computing sparsity
                    if "unstructured" in weight_path:
                        sparsity = extract_sparsity(weight_path)
                        print(weight_path, sparsity)
                    else:
                        sparsity = extract_density(weight_path)
                        print(weight_path, sparsity)
                    results[st][method_name][dataset]["sparsity"].append(sparsity)

                    # Computing Recall
                    if dataset == "pitts30k":
                        method.load_weights(weight_path)
                        rec = extract_recall(weight)
                    else:
                        method.load_weights(weight_path)
                        ds = get_dataset(dataset)
                        q_dl = ds.query_images_loader(
                            batch_size=32, preprocess=method.preprocess
                        )
                        method.compute_query_desc(q_dl)
                        m_dl = ds.map_images_loader(
                            batch_size=32, preprocess=method.preprocess
                        )
                        method.compute_map_desc(m_dl)
                        method.save_descriptors(ds.name)
                        eval = Eval(method, ds)
                        eval.compute_all_matches(1)
                        rec = eval.ratk(1)
                    results[st][method_name][dataset]["recall@1"].append(rec)

                    # computing non zero parameters
                    param_count = None
                    if isinstance(method.model, nn.Module):
                        param_count = sum(
                            torch.count_nonzero(param).item()
                            for param in method.model.parameters()
                        )
                    results[st][method_name][dataset]["param_count"].append(param_count)

                    # computing flops
                    flops = None
                    flops = measure_flops(method)
                    results[st][method_name][dataset]["flops"].append(flops)

                    # Computing Latency
                    # method = deploy_onnx_cpu(method)
                    # lat_cpu = measure_latency_cpu(method)
                    # results[st][method_name][dataset]["latency_cpu"].append(lat_cpu)

                    ## Computing Latency
                    # lat_gpu = None
                    # method = deploy_tensorrt_sparse(method)
                    # if torch.cuda.is_available():
                    #    lat_gpu = measure_latency_gpu(method)
                    # results[st][method_name][dataset]["latency_gpu"].append(lat_gpu)

                    print(
                        "sparsity:",
                        sparsity,
                        "parameters: ",
                        param_count,
                        "latency: ",
                        #    lat_cpu,
                    )


with open("data/results.pkl", "wb") as file:
    pickle.dump(results, file)
