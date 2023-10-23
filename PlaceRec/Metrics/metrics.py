import os
import time
from typing import Tuple, Union

import numpy as np
import onnx
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve
from thop import profile
from curves import pr_curve

from PlaceRec.Datasets import GardensPointWalking

from .curves import pr_curve


def recallatk(ground_truth: np.ndarray, similarity: np.ndarray, ground_truth_soft: Union[None, np.ndarray] = None, k: int = 1) -> float:
    """
    Computes the recall at rank k for given similarity matrix and ground truth.

    Parameters:
    - ground_truth (np.ndarray): Binary matrix representing the ground truth.
    - similarity (np.ndarray): Similarity matrix with same shape as ground_truth.
    - ground_truth_soft (np.ndarray, optional): Soft ground truth matrix. If given, its shape must match ground_truth.
    - k (int, optional): Rank at which recall is computed. Default is 1.

    Returns:
    - float: Recall at rank k.

    Raises:
    - AssertionError: If the shapes of the matrices don't match or if other constraints are not satisfied.
    """
    assert similarity.shape == ground_truth.shape, "S_in and GThard must have the same shape"
    if ground_truth_soft is not None:
        assert similarity.shape == ground_truth_soft.shape, "S_in and GTsoft must have the same shape"
    assert similarity.ndim == 2, "S_in, GThard and GTsoft must be two-dimensional"
    assert k >= 1, "K must be >=1"

    S = similarity.copy()
    if ground_truth_soft is not None:
        ground_truth_soft = ground_truth_soft.astype("bool")
        S[ground_truth_soft & ~ground_truth] = S.min()

    ground_truth = ground_truth.astype("bool")
    j = ground_truth.sum(0) > 0  # columns with matches
    S = S[:, j]  # select columns with a match
    ground_truth = ground_truth[:, j]  # select columns with a match
    i = S.argsort(0)[-k:, :]
    j = np.tile(np.arange(i.shape[1]), [k, 1])
    ground_truth = ground_truth[i, j]
    RatK = np.sum(ground_truth.sum(0) > 0) / ground_truth.shape[1]
    return RatK


def recall_at_100p(
    ground_truth: np.ndarray,
    similarity: np.ndarray,
    ground_truth_soft: Union[None, np.ndarray] = None,
    k: int = 1,
    matching: str = "mutli",
    n_thresh: int = 100,
) -> float:
    """
    Computes the recall at 100% precision for given similarity matrix and ground truth.

    Parameters:
    - ground_truth (np.ndarray): Binary matrix representing the ground truth.
    - similarity (np.ndarray): Similarity matrix.
    - ground_truth_soft (np.ndarray, optional): Soft ground truth matrix.
    - k (int, optional): Rank at which recall is computed. Default is 1.
    - matching (str, optional): Matching type, either "single" or "multi". Default is "multi".
    - n_thresh (int, optional): Number of thresholds. Default is 100.

    Returns:
    - float: Recall at 100% precision.

    Raises:
    - AssertionError: If the shapes of the matrices don't match or if other constraints are not satisfied.
    """

    assert similarity.shape == ground_truth.shape, "S_in and GThard must have the same shape"
    if ground_truth_soft is not None:
        assert similarity.shape == ground_truth_soft.shape, "S_in and GTsoft must have the same shape"
    assert similarity.ndim == 2, "S_in, GThard and GTsoft must be two-dimensional"
    assert n_thresh > 1, "n_thresh must be >1"

    # get precision-recall curve
    P, R = pr_curve(similarity=similarity, ground_truth=ground_truth, ground_truth_soft=ground_truth_soft, matching=matching, n_thresh=n_thresh)
    P = np.array(P)
    R = np.array(R)
    R = R[P == 1]
    R = R.max()
    return R


def precision(ground_truth: np.ndarray, preds: np.ndarray, ground_truth_soft: Union[None, np.ndarray] = None) -> float:
    """
    Computes the precision for given predictions and ground truth.

    Parameters:
    - ground_truth (np.ndarray): Binary matrix representing the ground truth.
    - preds (np.ndarray): Predictions matrix.
    - ground_truth_soft (np.ndarray, optional): Soft ground truth matrix.

    Returns:
    - float: Precision.

    Raises:
    - Exception: If there's a division by zero error.
    """

    if ground_truth_soft is not None:
        preds[~ground_truth & ground_truth_soft] = False
    preds = preds.astype(bool)
    ground_truth = ground_truth.astype(bool)
    TP = np.count_nonzero(ground_truth & preds)
    FP = np.count_nonzero((~ground_truth) & preds)
    if TP + FP == 0:
        raise Exception("Divide by zero. TP: " + str(TP) + "  FP: " + str(FP))
    return TP / (TP + FP)


def recall(ground_truth: np.ndarray, preds: np.ndarray, ground_truth_soft: Union[None, np.ndarray] = None) -> float:
    """
    Computes the recall for given predictions and ground truth.

    Parameters:
    - ground_truth (np.ndarray): Binary matrix representing the ground truth.
    - preds (np.ndarray): Predictions matrix.
    - ground_truth_soft (np.ndarray, optional): Soft ground truth matrix.

    Returns:
    - float: Recall.

    Raises:
    - Exception: If there's a division by zero error.
    """

    if ground_truth_soft is not None:
        preds[~ground_truth & ground_truth_soft] = False
    preds = preds.astype(bool)
    ground_truth = ground_truth.astype(bool)
    TP = np.count_nonzero(ground_truth & preds)
    GTP = np.count_nonzero(ground_truth)
    if GTP == 0:
        raise Exception("Divide by zero. GTP: 0")
    return TP / GTP


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
    assert isinstance(method.module, nn.module)
    if method.model is not None:
        for batch in loader:
            input = batch[0][None, :].to(method.device)  # get one input item
            flops, _ = profile(method.model, inputs=(input,))
            return int(flops)
    else:
        return 0


def count_params(method) -> int:
    """
    Counts the total number of parameters for the model in a given method.

    Parameters:
    - method: Method object which contains a model.

    Returns:
    - int: Total number of parameters.
    """

    assert isinstance(method.model, nn.module)
    if method.model is not None:
        total_params = sum(p.numel() for p in method.model.parameters())
        return int(total_params)
    else:
        return 0


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


def get_model_size_in_bytes(model, model_type):
    """
    Get the size of the given model in bytes.

    Parameters:
    - model: The neural network model (PyTorch, ONNX, or TensorRT).
    - model_type: A string indicating the type of the model ("pytorch", "onnx", or "tensorrt").

    Returns:
    - Size of the model in bytes.
    """

    temp_file = "/mnt/data/temp_model"

    if model_type == "pytorch":
        torch.save(model.state_dict(), temp_file)
    elif model_type == "onnx":
        with open(temp_file, "wb") as f:
            f.write(model.SerializeToString())
    elif model_type == "tensorrt":
        with open(temp_file, "wb") as f:
            f.write(model.serialize())
    else:
        raise ValueError("Unsupported model_type. Choose from 'pytorch', 'onnx', or 'tensorrt'.")

    # Get the file size in bytes
    size_in_bytes = os.path.getsize(temp_file)

    # Clean up the temporary file
    os.remove(temp_file)

    return size_in_bytes


def average_precision(ground_truth: np.ndarray, similarity: np.ndarray, ground_truth_soft: Union[None, np.ndarray] = None) -> float:
    """
    Compute the average precision (AP) for the given ground truth and similarity scores.

    The function first computes the precision-recall curve using the provided ground truth and similarity scores.
    It then calculates the average precision by summing the areas of rectangles under the precision-recall curve.

    Parameters:
    - ground_truth (np.ndarray): A 2D binary array where each column corresponds to a data point and each row
                                 corresponds to a possible match. An entry of 1 indicates a correct match, and 0 otherwise.
    - similarity (np.ndarray): A 2D array of the same shape as ground_truth, representing the similarity scores
                               between data points and possible matches. Higher values indicate higher similarity.
    - ground_truth_soft (Union[None, np.ndarray], optional): A 2D binary array of the same shape as ground_truth,
                                                            indicating soft ground truth values. An entry of 1 suggests
                                                            a possible match, but with lower confidence than the hard ground truth.
                                                            Default is None.

    Returns:
    - float: The average precision value, representing the area under the precision-recall curve.

    """
    P, R = pr_curve(ground_truth=ground_truth, similarity=similarity, ground_truth_soft=ground_truth_soft)
    # Ensure that recall is monotonically increasing
    for i in range(len(R) - 1, 0, -1):
        P[i - 1] = max(P[i - 1], P[i])

    # Compute the AP using the formula
    AP = 0.0
    for i in range(1, len(R)):
        AP += (R[i] - R[i - 1]) * P[i]
    return AP
