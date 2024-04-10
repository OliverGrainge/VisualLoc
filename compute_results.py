# from PlaceRec import Methods, Datasets
import os
import sys

import pickle
import re
from os.path import join

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from PlaceRec.Evaluate import Eval
from PlaceRec.utils import get_method, get_dataset
from torchprofile import profile_macs
from collections import defaultdict


CheckpointDirectory = "/Users/olivergrainge/Downloads/Checkpoints"
METHODS = ["cct_cls", "vit_cls", "mixvpr", "convap"]
DATASETS = ["pitts30k"]


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
def measure_latency_cpu(method, num_runs=100):
    img = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
    img = Image.fromarray(img)
    img = method.preprocess(img)
    img = img.to("cpu")
    if isinstance(method.model, nn.Module):
        method.set_device("cpu")

    for _ in range(10):
        method.model(img[None, :])
    st = time.time()
    for _ in tqdm(range(num_runs), desc=f"CPU Latency {method.name}"):
        method.model(img[None, :])
    et = time.time()
    return (et - st) / (num_runs) * 1000


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
            _ = method.model(img[None, :])
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    elapsed_time = 0.0
    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc=f"GPU Latency {method.name}"):
            start_event.record()
            _ = method.model(img[None, :])
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time += start_event.elapsed_time(end_event)
    avg_time = elapsed_time / num_runs
    return avg_time


def measure_flops(method):
    img = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
    img = Image.fromarray(img)
    img = method.preprocess(img)

    flops = profile_macs(method.model, (img[None, :],))
    return flops


sparsity_type = {
    "unstructured": "gsv_cities_sparse_unstructured/",
    "semistructured": "gsv_cities_sparse_semistructured/",
    "structured": "gsv_cities_sparse_structured/",
}

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

                    # Computing Latency
                    lat_cpu = measure_latency_cpu(method)
                    results[st][method_name][dataset]["latency_cpu"].append(lat_cpu)

                    # Computing Latency
                    lat_gpu = None
                    if torch.cuda.is_available():
                        lat_gpu = measure_latency_gpu(method)
                    results[st][method_name][dataset]["latency_gpu"].append(lat_gpu)

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
                    print(
                        "sparsity:",
                        sparsity,
                        "parameters: ",
                        param_count,
                        "latency: ",
                        lat_cpu,
                    )


with open("data/results.pkl", "wb") as file:
    pickle.dump(results, file)
