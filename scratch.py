from PlaceRec.Methods import NetVLAD 
from PlaceRec.Datasets import ESSEX3IN1
import numpy as np

ds = ESSEX3IN1()
method = NetVLAD()

query_loader = ds.query_images_loader(partition="all", preprocess=method.preprocess)
map_loader = ds.map_images_loader(partition="all", preprocess=method.preprocess)
q_desc = method.compute_query_desc(dataloader=query_loader)
m_desc = method.compute_map_desc(dataloader=map_loader)

ground_truth = ds.ground_truth(partition="all")



def pr_curve(method, ground_truth: list, matching: str="single", n_thresh=100):
    """
    Return the precision and recalls over a number of thesholds
    """
    preds, dist = method.place_recognise(query_desc=method.query_desc, k=1)
    ground_truth = np.array([1 if set(p).intersection(set(gt)) else 0 for p, gt in zip(preds, ground_truth)])
    similarity = preds.flatten()
    ground_truth = ground_truth.astype("bool")

    similarity = similarity.copy()
    GTP = np.count_nonzero(ground_truth)
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

P, R = pr_curve(method, ground_truth)


import matplotlib.pyplot as plt

plt.plot(R, P)
plt.show()

def recallatk(method, ground_truth: list, k: int = 1) -> float:
    preds, dist = method.place_recognise(query_desc=method.query_desc, k=k)
    result = [1 if set(p).intersection(set(gt)) else 0 for p, gt in zip(preds, ground_truth)]
    return np.mean(result)