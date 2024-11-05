import os
import pickle
import time
import warnings
from typing import Dict, List, Union

import faiss
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image
from ptflops import get_model_complexity_info
from tabulate import tabulate

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
    ):
        """
        Initializes the Eval class with a recognition method and a dataset.

        Args:
            method (BaseTechnique): The recognition technique to be evaluated.
            dataset (Union[BaseTechnique, None], optional): The dataset to be used for evaluation. Defaults to None.
        """
        self.dataset = dataset
        self.method = method
        self.gt = dataset.ground_truth()
        self.results = {}


    def convert_to_onnx(self):
        """
        Converts the PyTorch model to an ONNX model and saves it to the specified path.

        Args:
            input_size (tuple): The size of the input tensor (e.g., (1, 3, 224, 224) for a single RGB image).
            onnx_file_path (str): The file path where the ONNX model will be saved.

        Returns:
            None
        """

        if not isinstance(self.method.model, nn.Module):
            raise ValueError("The model must be a PyTorch nn.Module to convert to ONNX.")

        model = self.method.model
        model.eval()

        dummy_input = self.method.example_input()
        model = model.to("cpu")

        try:
            torch.onnx.export(
                model,
                dummy_input,
                "/tmp/model.onnx",
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
        except Exception as e:
            logger.error(f"Error converting model to ONNX: {e}")


    def setup_onnx_session_cpu(self):
        """
        Sets up an ONNX Runtime inference session for CPU execution.

        This method configures the session options for sequential execution mode
        and initializes an inference session using the CPUExecutionProvider.

        Returns:
            ort.InferenceSession: An ONNX Runtime inference session configured for CPU execution.
        """
        sess_options = ort.SessionOptions()
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            "/tmp/model.onnx", sess_options, providers=["CPUExecutionProvider"]
        )
        return session

    def setup_onnx_session_gpu(self):
        """
        Sets up an ONNX Runtime inference session for GPU execution.

        This method configures the session options for sequential execution mode
        and extended graph optimization level, and initializes an inference session
        using the CUDAExecutionProvider.

        Returns:
            ort.InferenceSession: An ONNX Runtime inference session configured for GPU execution.
        """
        sess_options = ort.SessionOptions()
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            session = ort.InferenceSession(
                "/tmp/model.onnx", sess_options, providers=["CUDAExecutionProvider"]
            )
        elif "CoreMLExecutionProvider" in available_providers:
            session = ort.InferenceSession(
                "/tmp/model.onnx", sess_options, providers=["CoreMLExecutionProvider"]
            )
        else:
            return None 
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
        self.convert_to_onnx()

        self.extraction_cpu_latency()
        self.extraction_gpu_latency()
        self.matching_latency()
        self.count_params()
        table_data = [(k, v) for k, v in self.results.items()]
        print(tabulate(table_data, headers=["Metric", "Value"]))
        return self.results

    def compute_all_matches(self, k=1):
        """
        Attempts to load descriptors and compute place recognition matches up to rank k.

        Args:
            k (int, optional): The rank limit for match computation. Defaults to 20.

        Returns:
            None: On failure, returns None and does not modify results.
        """
        self.method.load_descriptors(self.dataset.name)
        if self.method.query_desc is None:
            raise Exception("Query descriptors not loaded. You must pre-compute them before calling this method.")
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
        """
        Returns the dimensionality of the global descriptors used in the method.

        Returns:
            float: The number of dimensions of the global descriptors.
        """
        self.method.load_descriptors(self.dataset.name)
        return self.method.query_desc["global_descriptors"].shape[1]

    def map_memory(self) -> float:
        """
        Calculates and returns the memory usage of the map descriptors in megabytes.

        Returns:
            float: The memory usage of the map descriptors in MB.
        """
        self.method.load_descriptors(self.dataset.name)
        return self.method.map_desc["global_descriptors"].nbytes / (1024**2)

    def model_memory(self) -> float:
        """
        Estimates and returns the memory usage of the model parameters in megabytes,
        assuming the parameters are stored in fp16 precision.

        Returns:
            float: The memory usage of the model parameters in MB.
        """
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
        input_data = self.method.example_input()
        input_data = input_data.to("cpu")
        input_data = input_data.numpy()

        session = self.setup_onnx_session_cpu()

        for _ in range(10):
            out = session.run(None, {"input": input_data})
            self.desc_size = out[0].shape[1]

        # Measure inference time
        start_time = time.time()
        for _ in range(num_runs):
            _ = session.run(None, {"input": input_data})
        end_time = time.time()
        average_time = (
            (end_time - start_time) / num_runs
        ) * 1000  # Convert to milliseconds
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
        img = Image.fromarray(
            np.random.randint(0, 244, (224, 224, 3)).astype(np.uint8)
        )
        input_data = self.method.preprocess(img)[None, :]
        input_data = input_data.repeat(batch_size, 1, 1, 1)
        input_data = input_data.to("cpu")
        input_data = input_data.numpy()

        session = self.setup_onnx_session_gpu()

        if session is None: 
            return

        for _ in range(10):
            _ = session.run(None, {"input": input_data})

        # Measure inference time
        start_time = time.time()
        for _ in range(num_runs):
            _ = session.run(None, {"input": input_data})
        end_time = time.time()
        average_time = (
            (end_time - start_time) / num_runs * 1000
        )  # Convert to milliseconds

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
