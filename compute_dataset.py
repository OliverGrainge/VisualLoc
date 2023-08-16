from PlaceRec.Metrics import recallatk
from PlaceRec.utils import get_method, get_dataset
from sklearn.metrics import fbeta_score, average_precision_score
import pandas as pd
import numpy as np
from typing import Union
import os
from multiplexVPR import METHODS


# =========================== Selection Dataset Configuration =================
DATASETS = ["nordlands"]
PARTITION = "train"
METRICS = ["recall@1", "recall@3", "recall@5", "recall@10"]
# ============================================================================

def vect_recallatk(ground_truth: np.ndarray, similarity: np.ndarray, ground_truth_soft: Union[None, np.ndarray] = None,
              k: int=1) -> float:

    assert (similarity.shape == ground_truth.shape),"S_in and GThard must have the same shape"
    if ground_truth_soft is not None:
        assert (similarity.shape == ground_truth_soft.shape),"S_in and GTsoft must have the same shape"
    assert (similarity.ndim == 2),"S_in, GThard and GTsoft must be two-dimensional"
    assert (k >= 1),"K must be >=1"

    S = similarity.copy()
    if ground_truth_soft is not None:
            ground_truth_soft = ground_truth_soft.astype('bool')
            S[ground_truth_soft & ~ground_truth] = S.min()

    ground_truth = ground_truth.astype('bool')
    j = ground_truth.sum(0) > 0 # columns with matches
    S = S[:,j] # select columns with a match
    ground_truth = ground_truth[:,j] # select columns with a match
    i = S.argsort(0)[-k:,:]
    j = np.tile(np.arange(i.shape[1]), [k, 1])
    ground_truth = ground_truth[i, j]
    RatK = ground_truth.sum(0) > 0
    return RatK.astype(np.float32)



def compute_metric(ground_truth: np.ndarray, ground_truth_soft: np.ndarray, similarity: np.ndarray , metric_name: str):
    if metric_name == "recall@1":
        return vect_recallatk(ground_truth=ground_truth, ground_truth_soft=ground_truth_soft, similarity=similarity, k=1)

    elif metric_name == "recall@3":
        return vect_recallatk(ground_truth=ground_truth, ground_truth_soft=ground_truth_soft, similarity=similarity, k=3)

    elif metric_name == "recall@5":
        return vect_recallatk(ground_truth=ground_truth, ground_truth_soft=ground_truth_soft, similarity=similarity, k=5)

    elif metric_name == "recall@10":
        return vect_recallatk(ground_truth=ground_truth, ground_truth_soft=ground_truth_soft, similarity=similarity, k=10)



def compute_datasets():
    # setup the datasets
    datasets = [{} for _ in range(len(METRICS))]
    for dataset in datasets:
        dataset["query_images"] = []
        for method_name in METHODS:
            dataset[method_name] = []
    
    # compute the data
    for dataset_name in DATASETS:
        ds = get_dataset(dataset_name)
        for dataset in datasets:
            dataset["query_images"] += ds.query_partition(partition=PARTITION).tolist()
        
        ground_truth = ds.ground_truth(partition=PARTITION, gt_type="hard")
        ground_truth_soft = ds.ground_truth(partition=PARTITION, gt_type="soft")
        for method_name in METHODS: 
            method = get_method(method_name)
            query_loader = ds.query_images_loader(partition=PARTITION, shuffle=False, num_workers=16, preprocess=method.preprocess, batch_size=64)
            map_loader = ds.map_images_loader(partition=PARTITION, shuffle=False, num_workers=16, preprocess=method.preprocess, batch_size=64)
            query_desc = method.compute_query_desc(dataloader=query_loader)
            map_desc = method.compute_map_desc(dataloader=map_loader)
            similarity = method.similarity_matrix(query_desc, map_desc)

            scores = [compute_metric(ground_truth=ground_truth,
                                     ground_truth_soft=ground_truth_soft,
                                     similarity=similarity, 
                                     metric_name=metric_name) for metric_name in METRICS]
            
            for i, score in enumerate(scores):
                datasets[i][method_name] += score.tolist()

            del method 
            del query_loader
            del map_loader
        
    pd_datasets = [pd.DataFrame.from_dict(dataset) for dataset in datasets]
    return pd_datasets
                            




def save_datasets(pd_datasets: list):
    for i, df in enumerate(pd_datasets):
        name = "_".join(DATASETS) + "_" + METRICS[i] + "_" + PARTITION + ".csv"
        df.to_csv(os.getcwd() + '/SelectionData/' + name)


def summarise_datasets(pd_datasets: list):
    for i, df in enumerate(pd_datasets):
        print(" ")
        print("=============================", METRICS[i], "======================================")
        print(df.describe())




if __name__ == "__main__":
    pd_datasets = compute_datasets()
    save_datasets(pd_datasets)
    summarise_datasets(pd_datasets)
