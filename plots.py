# from PlaceRec import Methods, Datasets
import os
import sys

sys.path.append(os.path.abspath("../"))
import pickle
import re
from os.path import join
from PlaceRec.Evaluate import Eval
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from PlaceRec.utils import get_method, get_dataset

sparsity_type = {
    "unstructured": "gsv_cities_sparse_unstructured/",
    "semistructured": "gsv_cities_sparse_semistructured/",
    "structured": "gsv_cities_sparse_structured/",
}

with open("data/results.pkl", "rb") as file:
    results = pickle.load(file)
print(list(results["unstructured"]["cct_cls"]["pitts30k"].keys()))

# ====================== Sparsity vs Recall@1 ======================
for st in sparsity_type.keys():
    if st == "semistructured":
        continue
    m = list(results[st].keys())[0]
    for ds in results[st][m].keys():
        for method in results[st].keys():
            sparsity = results[st][method][ds]["sparsity"]
            recall = results[st][method][ds]["recall@1"]
            sorted_pairs = sorted(zip(sparsity, recall), key=lambda x: x[0])
            sorted_sparsity, sorted_recall = zip(*sorted_pairs)
            sorted_sparsity = list(sorted_sparsity)
            sorted_latency = list(sorted_recall)
            plt.plot(sorted_sparsity, sorted_recall, label=method)
        plt.legend()
        plt.title(st + f" sparsity vs {ds} R@1")
        plt.xlabel("sparsity (%)")
        plt.ylabel("recall@1 (%)")
        plt.show()


# ===================== CPU latency vs Recall@1 =======================
for st in sparsity_type.keys():
    if st == "semistructured":
        continue
    m = list(results[st].keys())[0]
    for ds in results[st][m].keys():
        for method in results[st].keys():
            recall = results[st][method][ds]["recall@1"]
            lat = results[st][method][ds]["latency_cpu"]
            sparsity = results[st][method][ds]["sparsity"]
            assert len(recall) > 0
            assert len(lat) > 0
            assert len(sparsity) > 0
            sorted_pairs = sorted(zip(sparsity, lat, recall), key=lambda x: x[0])
            sorted_sparsity, sorted_lat, sorted_recall = zip(*sorted_pairs)
            sorted_lat = list(sorted_lat)
            sorted_recall = list(sorted_recall)
            plt.plot(sorted_lat, sorted_recall, label=method)
        plt.legend()
        plt.title(st + f" sparsity cpu latency vs {ds} R@1")
        plt.xlabel("latency (ms)")
        plt.ylabel("recall@1 (%)")
        plt.show()


# ===================== GPU latency vs Recall@1 =======================
break_flag = False
for st in sparsity_type.keys():
    if st == "semistructured":
        continue
    m = list(results[st].keys())[0]
    for ds in results[st][m].keys():
        for method in results[st].keys():
            recall = results[st][method][ds]["recall@1"]
            lat = results[st][method][ds]["latency_gpu"]
            sparsity = results[st][method][ds]["sparsity"]
            if None in lat:
                break_flag = True
                break
            sorted_pairs = sorted(zip(sparsity, lat, recall), key=lambda x: x[0])
            sorted_sparsity, sorted_lat, sorted_recall = zip(*sorted_pairs)
            sorted_lat = list(sorted_lat)
            sorted_latency = list(sorted_recall)
            plt.plot(sorted_lat, sorted_recall, label=method)
        if break_flag:
            break
        plt.legend()
        plt.title(st + f"sparse gpu latency vs {ds} R@1")
        plt.xlabel("gpu latency (ms)")
        plt.ylabel("recall@1 (%)")
        plt.show()


# ===================== Non zero parameters vs Recall@1 =======================
for st in sparsity_type.keys():
    if st == "semistructured":
        continue
    m = list(results[st].keys())[0]
    for ds in results[st][m].keys():
        for method in results[st].keys():
            params = results[st][method][ds]["param_count"]
            recall = results[st][method][ds]["recall@1"]
            sparsity = results[st][method][ds]["sparsity"]
            sorted_pairs = sorted(zip(sparsity, params, recall), key=lambda x: x[0])
            sorted_sparsity, sorted_params, sorted_recall = zip(*sorted_pairs)
            sorted_params = list(sorted_params)
            sorted_recall = list(sorted_recall)
            plt.plot(sorted_params, sorted_recall, label=method)
        plt.legend()
        plt.title(st + f" sparse param count vs {ds} R@1")
        plt.xlabel("non zero parameter count")
        plt.ylabel("recall@1 (%)")
        plt.show()
