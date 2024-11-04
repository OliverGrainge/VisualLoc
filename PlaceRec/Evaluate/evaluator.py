import os
import time
import warnings
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from ptflops import get_model_complexity_info
from tabulate import tabulate
import onnxruntime as ort
import faiss
import pickle
from PlaceRec.Methods.base_method import BaseTechnique
from PlaceRec.utils import get_logger

logger = get_logger()


class Eval:
    """
    A class to evaluate various performance metrics for image recognition methods
    on a specified dataset.

    Attributes:
        method (BaseTechnique): The recognition technique to be evaluated.
        dataset (Union[BaseTechnique, None]): The dataset on which the technique is evaluated.
        results (dict): A dictionary to store evaluation results.
        gt (list): Ground truth data for the dataset, loaded if dataset is not None.

    Methods:
        eval(): Orchestrates the evaluation by running all metrics.
        compute_all_matches(): Computes matches for recognition up to a specified rank.
        ratk(k): Computes and returns the recall at rank k.
        matching_latency(): Measures and returns the average time taken to match queries.
        extraction_cpu_latency(): Measures and returns the CPU latency for feature extraction.
        extraction_gpu_latency(): Measures and returns the GPU latency for feature extraction.
        count_params(): Returns the number of parameters in the model.
        count_flops(): Estimates and returns the FLOPS of the model.
    """

    def __init__(
        self,
        method: BaseTechnique,
        dataset: Union[BaseTechnique, None] = None,
        onnx_pth=None,
    ):
        """
        Initializes the Eval class with a recognition method and a dataset.

        Args:
            method (BaseTechnique): The recognition technique to be evaluated.
            dataset (Union[BaseTechnique, None], optional): The dataset to be used for evaluation. Defaults to None.
        """
        self.dataset = dataset
        self.method = method
        self.onnx_pth = onnx_pth
        if onnx_pth is None:
            self.gt = dataset.ground_truth()

        self.results = {}


    def setup_onnx_session_cpu(self):
        sess_options = ort.SessionOptions()
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            self.onnx_pth, sess_options, providers=["CPUExecutionProvider"]
        )
        return session

    def setup_onnx_session_gpu(self):
        sess_options = ort.SessionOptions()
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        session = ort.InferenceSession(
            self.onnx_pth, sess_options, providers=["CUDAExecutionProvider"]
        )
        return session

    def eval(self):
        """
        Executes all evaluation metrics and stores results in the results dictionary.

        Returns:
            Dict: A dictionary containing all computed evaluation metrics.
        """
        
        if self.dataset is not None:
            self.compute_all_matches()
            self.ratk(1)
            self.ratk(5)
            self.ratk(10)
            self.ratk(20)
        self.extraction_cpu_latency()
        self.extraction_gpu_latency()
        self.matching_latency()
        self.count_params()
        table_data = [(k, v) for k, v in self.results.items()]
        print(tabulate(table_data, headers=["Metric", "Value"]))
        return self.results

    def compute_all_matches(self, k=20):
        """
        Attempts to load descriptors and compute place recognition matches up to rank k.

        Args:
            k (int, optional): The rank limit for match computation. Defaults to 20.

        Returns:
            None: On failure, returns None and does not modify results.
        """
        print("hi")
        self.method.load_descriptors(self.dataset.name)
        self.matches, self.distances = self.method.place_recognise(
            self.method.query_desc, k=k
        )
        

    def ratk(self, k: int) -> Dict:
        """
        Computes the recall at rank k for the loaded matches and updates the results dictionary.

        Args:
            k (int): The rank at which recall is calculated.

        Returns:
            float: The computed recall at rank k, or None if descriptors are not properly loaded.
        """
        self.method.load_descriptors(self.dataset.name)
        if self.method.query_desc == None:
            return None
        elif self.method.map_desc == None:
            return None
        kmatches = self.matches[:, :k]
        result = [
            1 if set(p).intersection(set(gt)) else 0 for p, gt in zip(kmatches, self.gt)
        ]
        ratk = np.mean(result).astype(np.float32)
        self.results[f"r@{k}_{self.dataset.name}"] = ratk
        return ratk

    def descriptor_dim(self) -> float:
        self.method.load_descriptors(self.dataset.name)
        return self.method.query_desc["global_descriptors"].shape[1]

    def map_memory(self) -> float:
        self.method.load_descriptors(self.dataset.name)
        return self.method.map_desc["global_descriptors"].nbytes / (1024**2)

    def model_memory(self) -> float:
        params = self.count_params()
        return (params * 2) / (1024**2)

    def matching_latency(self, num_runs: int = 20) -> float:
        """
        Measures and returns the average latency of the place recognition method over a specified number of runs.

        Args:
            num_runs (int): The number of times to run the recognition process to average the latency.

        Returns:
            float: The average matching latency in milliseconds.
        """
        if isinstance(self.dataset, str):
            with open("dataset_sizes.pkl", "rb") as f:
                dataset_dict = pickle.load(f)
            dataset_size = dataset_dict[self.dataset]
            db_vectors = np.random.random((dataset_size, self.desc_size)).astype(
                "float32"
            )
            db_vectors /= np.linalg.norm(db_vectors, axis=1)[:, np.newaxis]
            index = faiss.IndexFlatIP(self.desc_size)  # Inner Product (IP) index
            index.add(db_vectors)
            times = []
            query_vectors = np.random.random((1, self.desc_size)).astype(np.float32)
            for _ in range(num_runs):
                st = time.time()
                D, I = index.search(query_vectors, 1)  # D: distances, I: indices
                et = time.time()
                times.append(et - st)
            return np.mean(times) * 1000

        self.method.load_descriptors(self.dataset.name)
        single_query_desc = {}
        for key, value in self.method.query_desc.items():
            single_query_desc[key] = value[0][None, :]

        st = time.time()
        for _ in range(num_runs):
            self.matches, self.distances = self.method.place_recognise(
                single_query_desc, k=20
            )
        et = time.time()

        self.results[f"{self.dataset.name}_matching_latency_ms"] = (
            (et - st) / num_runs
        ) * 1000

        return ((et - st) / num_runs) * 1000

    def extraction_cpu_latency(self, batch_size: int = 1, num_runs: int = 100) -> float:
        """
        Measures and returns the CPU latency of the feature extraction process for the model over a number of runs.

        Args:
            num_runs (int): The number of times the feature extraction is run to calculate the average latency.

        Returns:
            float: The average CPU extraction latency in milliseconds.
        """
        model = self.method.model
        model.eval()
        model = model.cpu()
        img = Image.fromarray(np.random.randint(0, 244, (224, 224, 3)).astype(np.uint8))
        input_data = self.method.preprocess(img)[None, :]
        input_data = input_data.repeat(batch_size, 1, 1, 1)

        if self.onnx_pth is not None:
            input_data = input_data.to("cpu")
            input_data = input_data.numpy()

            session = self.setup_onnx_session_cpu()

            for _ in range(10):
                out = session.run(None, {"input.1": input_data})
                self.desc_size = out[0].shape[1]

            # Measure inference time
            start_time = time.time()
            for _ in range(num_runs):
                _ = session.run(None, {"input.1": input_data})
            end_time = time.time()
            average_time = (
                (end_time - start_time) / num_runs
            ) * 1000  # Convert to milliseconds
            self.results["extraction_cpu_latency_ms"] = average_time
            return average_time

        else:

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
            self.results["extraction_cpu_latency_ms"] = average_time
            return average_time

    def extraction_gpu_latency(self, batch_size: int = 1, num_runs: int = 100) -> float:
        """
        Measures and returns the GPU latency of the feature extraction process for the model over a number of runs.

        Args:
            num_runs (int): The number of times the feature extraction is run on the GPU to calculate the average latency.

        Returns:
            float: The average GPU extraction latency in milliseconds, or None if CUDA is not available.
        """
        if self.onnx_pth is not None:
            img = Image.fromarray(
                np.random.randint(0, 244, (224, 224, 3)).astype(np.uint8)
            )
            input_data = self.method.preprocess(img)[None, :]
            input_data = input_data.repeat(batch_size, 1, 1, 1)
            input_data = input_data.to("cpu")
            input_data = input_data.numpy()

            session = self.setup_onnx_session_gpu()

            for _ in range(10):
                _ = session.run(None, {"input.1": input_data})

            # Measure inference time
            start_time = time.time()
            for _ in range(num_runs):
                _ = session.run(None, {"input.1": input_data})
            end_time = time.time()
            average_time = (
                (end_time - start_time) / num_runs * 1000
            )  # Convert to milliseconds

            self.results["extraction_gpu_latency_ms"] = average_time
            return average_time
        if not torch.cuda.is_available():
            return None
        model = self.method.model
        model.cuda()
        model.eval()
        img = Image.fromarray(np.random.randint(0, 244, (224, 224, 3)).astype(np.uint8))
        input_data = self.method.preprocess(img)[None, :].cuda()
        input_data = input_data.repeat(batch_size, 1, 1, 1)
        for _ in range(10):
            _ = model(input_data)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_runs):
            _ = model(input_data)
        end_event.record()
        torch.cuda.synchronize()
        average_time = start_event.elapsed_time(end_event) / num_runs
        self.results["extraction_gpu_latency_ms"] = average_time
        return average_time

    def count_params(self) -> int:
        """
        Counts and returns the total number of trainable parameters in the model.

        Returns:
            int: The total number of parameters, or None if the model is not an instance of nn.Module.
        """
        if not isinstance(self.method.model, nn.Module):
            warnings.warn(
                "Evaluator cannot compute nparams: method.model is not of type nn.Module"
            )
            return None
        total_params = sum(p.numel() for p in self.method.model.parameters())
        self.results["nparams"] = total_params
        return int(total_params)

    def count_flops(self) -> int:
        """
        Estimates and returns the floating-point operations per second (FLOPS) of the model during inference.

        Returns:
            int: The number of FLOPS, or logs an informational message if computation fails.
        """
        try:
            self.method.model.eval()
            img = Image.fromarray(
                np.random.randint(0, 244, (224, 224, 3)).astype(np.uint8)
            )
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
        except:
            logger.info(f"Could not compute flops for {self.method.name}")
            return None
