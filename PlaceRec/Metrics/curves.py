import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def pr_curve(method, ground_truth: list, n_thresh=100):
    """
    Return the precision and recalls over a number of thesholds
    """
    preds, dist = method.place_recognise(query_desc=method.query_desc, k=1)
    ground_truth = np.array(
        [1 if set(p).intersection(set(gt)) else 0 for p, gt in zip(preds, ground_truth)]
    )
    similarity = preds.flatten()
    ground_truth = ground_truth.astype("bool")

    similarity = similarity.copy()
    GTP = np.count_nonzero(ground_truth)
    if GTP == 0:
        return np.zeros(10), np.linspace(0, 1, 10)
    R = [
        0,
    ]
    P = [
        1,
    ]
    startV = similarity.max()  # start-value for threshold
    endV = similarity.min()  # end-value for treshold
    for i in np.linspace(startV, endV, n_thresh):
        B = similarity >= i  # apply thresh
        TP = np.count_nonzero(ground_truth & B)  # true positives
        FP = np.count_nonzero((~ground_truth) & B)  # false positives
        P.append(TP / (TP + FP))  # precision
        R.append(TP / GTP)  # recall
    return np.array(P), np.array(R)


def plot_pr_curve(
    methods: list,
    ground_truth: list,
    n_thresh: int = 100,
    title: str = None,
    show: bool = True,
    dataset_name: str = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Plot of multiple pr curves on a single graph. all_similarity is a dictionary
    of keys consisting of method names and values the similarity matrix it produced.
    """
    fig, ax = plt.subplots()
    for method in methods:
        P, R = pr_curve(method, ground_truth, n_thresh=n_thresh)
        ax.plot(R, P, label=method.name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    if title:
        ax.set_title(title)
    else:
        ax.set_title("PR Curve")
    if dataset_name is not None:
        pth = os.getcwd() + "/Plots/PlaceRec/prcurve"
        if not os.path.exists(pth):
            os.makedirs(pth)
        fig.savefig(pth + "/" + dataset_name + ".png")
    if show == True:
        plt.show()
    return ax


def plot_metric(
    methods: list,
    scores: np.ndarray,
    dataset_name: str,
    title: str,
    show: bool = True,
    metric_name="no_name_given",
):
    fig, ax = plt.subplots()
    bars = ax.bar(methods, scores)
    plt.xticks(rotation=45)
    ax.set_title(title, fontsize="16")
    plt.ylim([0, max(scores) + 0.2])
    plt.subplots_adjust(bottom=0.2)

    # Add legend as text annotation to each bar
    for bar, method, score in zip(bars, methods, scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{method}={score:.2f}",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=8,
        )

    pth = os.getcwd() + "/Plots/PlaceRec/" + metric_name
    if not os.path.exists(pth):
        os.makedirs(pth)
    fig.savefig(pth + "/" + dataset_name + ".png")
    if show:
        plt.show()
    return ax
