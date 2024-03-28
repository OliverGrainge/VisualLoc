from PlaceRec import Metrics
import numpy as np
from PlaceRec.Methods.base_method import BaseTechnique
from typing import List, Union, Dict
import time
import torch
import warnings
import torch.nn as nn
from ptflops import get_model_complexity_info
from PIL import Image


class Eval:
    def __init__(
        self, method: BaseTechnique, dataset: Union[BaseTechnique, None] = None
    ):
        self.dataset = dataset
        self.method = method
        if self.dataset is not None:
            self.gt = dataset.ground_truth()

        self.results = {}

    def eval_all(self):
        self.compute_all_matches()
        self.ratk(1)
        self.ratk(5)
        self.ratk(10)
        self.ratk(20)
        self.cpu_latency()
        self.gpu_latency()
        self.count_params()
        self.count_flops()
        return self.results

    def eval_performance(self):
        self.cpu_latency()
        self.gpu_latency()
        self.count_params()
        self.count_flops()
        return self.results

    def eval_accuracy(self):
        self.compute_all_matches()
        self.ratk(1)
        self.ratk(5)
        self.ratk(10)
        self.ratk(20)
        return self.results

    def compute_all_matches(self, k=20):
        try:
            self.method.load_descriptors(self.dataset.name)
            self.matches, self.distances = self.method.place_recognise(
                self.method.query_desc, k=k
            )
        except:
            return None

    def ratk(self, k: int) -> Dict:
        self.method.load_descriptors(self.dataset.name)
        if self.method.query_desc == None:
            warnings.warn("Query descriptors are not pre-computed")
            return None
        elif self.method.map_desc == None:
            warnings.warn("Map Descriptors are not pre-computed")
            return None
        kmatches = self.matches[:, :k]
        result = [
            1 if set(p).intersection(set(gt)) else 0 for p, gt in zip(kmatches, self.gt)
        ]
        ratk = np.mean(result).astype(np.float32)
        self.results[f"r@{k}_{self.dataset.name}"] = ratk
        return ratk

    def cpu_latency(self, num_runs: int = 100) -> float:
        model = self.method.model
        model.eval()
        model = model.cpu()
        img = Image.fromarray(np.random.randint(0, 244, (224, 224, 3)).astype(np.uint8))
        input_data = self.method.preprocess(img)[None, :]
        for _ in range(10):
            _ = model(input_data)

        # Measure inference time
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(input_data)
        end_time = time.time()
        average_time = (
            (end_time - start_time) / num_runs * 1000
        )  # Convert to milliseconds
        self.results["cpu_latency_ms"] = average_time
        return average_time

    def gpu_latency(self, num_runs: int = 100) -> float:
        if not torch.cuda.is_available():
            warnings.warn("Cuda is not available: Cannot evaluate gpu latency")
            return None
        model = self.method.model
        model.cuda()
        model.eval()
        img = Image.fromarray(np.random.randint(0, 244, (224, 224, 3)).astype(np.uint8))
        input_data = self.method.preprocess(img)[None, :].cuda()
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
        self.results["gpu_latency_ms"] = average_time
        return average_time

    def count_params(self) -> int:
        if not isinstance(self.method.model, nn.Module):
            warnings.warn(
                "Evaluator cannot compute nparams: method.model is not of type nn.Module"
            )
            return None
        total_params = sum(p.numel() for p in self.method.model.parameters())
        self.results["nparams"] = total_params
        return int(total_params)

    def count_flops(self) -> int:
        self.method.model.eval()
        img = Image.fromarray(np.random.randint(0, 244, (224, 224, 3)).astype(np.uint8))
        img = self.method.preprocess(img)
        flops, params = get_model_complexity_info(
            self.method.model,
            tuple(img.shape),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        self.results["flops"] = flops
        return flops
