import os
import sys
import time
from typing import Tuple, Union

import numpy as np
import onnx
import torch
import torch.nn as nn
from pympler import asizeof
from sklearn.metrics import precision_recall_curve
from thop import profile

from PlaceRec.Datasets import GardensPointWalking

from .curves import pr_curve


def recallatk(method, ground_truth: list, k: int = 1) -> float:
    preds, _ = method.place_recognise(query_desc=method.query_desc, k=k)
    result = [
        1 if set(p).intersection(set(gt)) else 0 for p, gt in zip(preds, ground_truth)
    ]
    return np.mean(result).astype(np.float32)


def recall_at_100p(
    method,
    ground_truth: list,
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

    # get precision-recall curve
    P, R = pr_curve(method=method, ground_truth=ground_truth, n_thresh=n_thresh)
    P = np.array(P)
    R = np.array(R)
    R = R[P == 1]
    R = R.max()
    return R


def average_precision(method, ground_truth: list, n_thresh: int = 100) -> float:
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
    P, R = pr_curve(method=method, ground_truth=ground_truth, n_thresh=n_thresh)
    # Ensure that recall is monotonically increasing
    for i in range(len(R) - 1, 0, -1):
        P[i - 1] = max(P[i - 1], P[i])

    # Compute the AP using the formula
    AP = 0.0
    for i in range(1, len(R)):
        AP += (R[i] - R[i - 1]) * P[i]
    return AP
