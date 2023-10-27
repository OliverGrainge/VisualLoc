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
    matching: str = "single",
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
