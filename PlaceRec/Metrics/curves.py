import numpy as np
from typing import Union
import matplotlib.pyplot as plt
import os
from PIL import Image

def pr_curve(ground_truth:np.ndarray, similarity: np.ndarray, ground_truth_soft: Union[None, np.ndarray] = None, n_thresh: int=100,
             matching: str = 'multi'):
    """
    Return the precision and recalls over a number of thesholds
    """
    assert similarity.shape == ground_truth.shape, "S and GT must be the same shape"
    assert (similarity.ndim == 2), "S_in, GThard and GTsoft must be two-dimensional"
    ground_truth = ground_truth.astype('bool')
    similarity = similarity.copy()
    if ground_truth_soft is not None:
        similarity[ground_truth_soft & ~ground_truth] = similarity.min()
    # single-best-match or multi-match VPR
    if matching == 'single':
        GTP = np.count_nonzero(ground_truth.any(0))
        ground_truth = ground_truth[np.argmax(similarity, axis=0), np.arange(ground_truth.shape[1])]
        similarity = np.max(similarity, axis=0)
    elif matching == 'multi':
        GTP = np.count_nonzero(ground_truth)  # ground truth positive
    R = [0, ]
    P = [1, ]
    startV = similarity.max()  # start-value for threshold
    endV = similarity.min()  # end-value for treshold
    for i in np.linspace(startV, endV, n_thresh):
        B = similarity >= i  # apply thresh
        TP = np.count_nonzero(ground_truth & B)  # true positives
        FP = np.count_nonzero((~ground_truth) & B)  # false positives
        P.append(TP / (TP + FP))  # precision
        R.append(TP / GTP)  # recall
    return np.array(P), np.array(R)





def plot_pr_curve(ground_truth: np.ndarray, all_similarity: dict, ground_truth_soft: Union[None, np.ndarray] = None, n_thresh: int=100,
             matching: str = 'multi', title: str = None, show: bool=True, dataset_name: str=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Plot of multiple pr curves on a single graph. all_similarity is a dictionary
    of keys consisting of method names and values the similarity matrix it produced. 
    """
    fig, ax = plt.subplots()
    for method_name, similarity in all_similarity.items():
        P, R = pr_curve(ground_truth, similarity, ground_truth_soft, n_thresh=100)
        ax.plot(R, P, label=method_name)
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
    if show ==True:
        plt.show()
    return ax



def plot_metric(methods: list, scores: np.ndarray, dataset_name: str, title: str, show: bool=True, metric_name="no_name_given"):
    fig, ax = plt.subplots()
    ax.bar(methods, scores)
    plt.xticks(rotation=45)
    ax.set_title(title, fontsize='16')
    pth = os.getcwd() + "/Plots/PlaceRec/" + metric_name 
    if not os.path.exists(pth):
        os.makedirs(pth)
    fig.savefig(pth + "/" + dataset_name + ".png")
    if show:
        plt.show()
    return ax 



def plot_dataset_sample(dataset, gt_type: str, show: bool=True) -> None:
    n_matches = 3
    n_queries = 4
    fig, ax = plt.subplots(n_queries, n_matches + 1, figsize=(15, 8))
    query_paths = dataset.test_query_paths
    map_paths = dataset.test_map_paths
    ground_truth = dataset.ground_truth(partition="test", gt_type=gt_type)

    samples = np.random.randint(0, len(query_paths), n_queries)
    ref_matches = [np.argwhere(ground_truth[:, samp]) for samp in samples]
    ref_paths = [map_paths[ref_match].flatten() for ref_match in ref_matches]

    for i in range(n_queries):
        print("Query: ", query_paths[samples[i]])
        ax[i, 0].imshow(np.array(Image.open(query_paths[samples[i]])))
        ax[i, 0].axis('off')
        ax[0, 0].set_title("Query Images")
        for j in range(1, n_matches + 1):
            ax[i, j].axis('off')
            if i == 0:
                ax[i, j].set_title("Ref Image " + str(j))
            try:
                ax[i, j].imshow(np.array(Image.open(ref_paths[i][j-1])))
            except: 
                continue
    if show:
        plt.suptitle("Sample of " + dataset.name + " Dataset", fontsize="18")
        plt.tight_layout()
        plt.show()

    pth = os.getcwd() + "/Plots/Dataset_Samples/" 
    if not os.path.exists(pth):
        os.makedirs(pth)
    fig.savefig(pth + "/" + dataset.name + ".png")
    return None









