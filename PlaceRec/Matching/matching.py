import numpy as np
from scipy.stats import norm



def threshold(similarity: np.ndarray, threshold: float) -> np.ndarray:
    matches = similarity >= threshold
    return matches


def auto_threshold(similarity: np.ndarray) -> np.ndarray:
    mu = np.median(similarity)
    sig = np.median(np.abs(similarity - mu)) / 0.675
    thresh = norm.ppf(1 - 1e-6, loc=mu, scale=sig)
    matches = similarity >= thresh
    return matches


def single_match_threshold(similarity: np.ndarray) -> np.ndarray:
    i = np.argmax(similarity, axis=0)
    j = np.int64(range(len(i)))
    matches = np.zeros_like(similarity, dtype='bool')
    matches[i, j] = True
    return matches
