import numpy as np
from sklearn.metrics import precision_recall_curve
from .curves import pr_curve
from typing import Union, Tuple
from thop import profile

def recallatk(ground_truth: np.ndarray, similarity: np.ndarray, ground_truth_soft: Union[None, np.ndarray] = None,
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
    RatK = np.sum(ground_truth.sum(0) > 0) / ground_truth.shape[1]
    return RatK


def recall_at_100p(ground_truth: np.ndarray, similarity: np.ndarray, ground_truth_soft: Union[None, np.ndarray] = None,
              k: int=1, matching: str = "mutli", n_thresh: int=100) -> float:
    assert (similarity.shape == ground_truth.shape),"S_in and GThard must have the same shape"
    if ground_truth_soft is not None:
        assert (similarity.shape == ground_truth_soft.shape),"S_in and GTsoft must have the same shape"
    assert (similarity.ndim == 2),"S_in, GThard and GTsoft must be two-dimensional"
    assert (matching in ['single', 'multi']),"matching should contain one of the following strings: [single, multi]"
    assert (n_thresh > 1),"n_thresh must be >1"

    # get precision-recall curve
    P, R = pr_curve(similarity=similarity, ground_truth=ground_truth,
                    ground_truth_soft=ground_truth_soft, matching=matching, n_thresh=n_thresh)
    P = np.array(P)
    R = np.array(R)
    R = R[P==1]
    R = R.max()
    return R

    

def precision(ground_truth: np.ndarray, preds: np.ndarray, ground_truth_soft: Union[None, np.ndarray] = None) -> float:
    if ground_truth_soft is not None:
        preds[~ground_truth & ground_truth_soft] = False
    preds = preds.astype(bool)
    ground_truth = ground_truth.astype(bool)
    TP = np.count_nonzero(ground_truth & preds)
    FP = np.count_nonzero((~ground_truth) & preds)
    if TP + FP == 0:
        raise Exception("Divide by zero. TP: " + str(TP) + "  FP: " + str(FP))
    return TP/(TP + FP)


def recall(ground_truth: np.ndarray, preds: np.ndarray, ground_truth_soft: Union[None, np.ndarray] = None) -> float:
    if ground_truth_soft is not None:
        preds[~ground_truth & ground_truth_soft] = False
    preds = preds.astype(bool)
    ground_truth = ground_truth.astype(bool)
    TP = np.count_nonzero(ground_truth & preds)
    GTP = np.count_nonzero(ground_truth)
    if GTP == 0:
        raise Exception("Divide by zero. GTP: 0")
    return TP/GTP


def count_flops(method) -> int:
    from PlaceRec.Datasets import GardensPointWalking
    ds = GardensPointWalking()
    loader = ds.query_images_loader("test", preprocess=method.preprocess)
    if method.model is not None:
        for batch in loader:
            input = batch[0][None, :].to(method.device) # get one input item 
            flops, _ = profile(method.model, inputs=(input,))
            return int(flops)
    else:
        return 0

def count_params(method) -> int:
    if method.model is not None:
        total_params = sum(p.numel() for p in method.model.parameters())
        return int(total_params)
    else:
        return 0






    

