import unittest

import numpy as np

from PlaceRec.Metrics import precision, recall, recallatk


class TestMetrics(unittest.TestCase):
    def test_precision(self):
        ground_truth = np.ones(shape=(10, 10)).astype(bool)
        ground_truth[:5, :] = False
        preds = np.zeros(shape=(10, 10)).astype(bool)
        preds[8:, :] = True

        prec = precision(ground_truth=ground_truth, preds=preds)

        assert prec == 1.0

        prec = precision(ground_truth=ground_truth, preds=preds, ground_truth_soft=ground_truth)

        assert prec == 1.0

        ground_truth_soft = np.ones(shape=(10, 10)).astype(bool)
        ground_truth_soft[:4, :] = False
        preds = np.zeros(shape=(10, 10)).astype(bool)
        preds[4:, :] = True

        prec = precision(ground_truth=ground_truth, preds=preds, ground_truth_soft=ground_truth_soft)
        assert prec == 1.0

        ground_tuth = np.ones(shape=(10, 10)).astype(bool)
        ground_truth[:5, :] = False
        preds = np.zeros(shape=(10, 10)).astype(bool)
        preds[:5, :] = True

        prec = precision(ground_truth=ground_truth, preds=preds)

        assert prec == 0.0

    def test_recall(self):
        ground_truth = np.ones(shape=(10, 10)).astype(bool)
        preds = np.ones(shape=(10, 10)).astype(bool)

        rec = recall(ground_truth=ground_truth, preds=preds)
        assert rec == 1.0

        preds = np.ones(shape=(10, 10)).astype(bool)
        preds[:5, :] = False

        rec = recall(ground_truth=ground_truth, preds=preds)
        assert rec == 0.5

        ground_truth = np.ones(shape=(10, 10)).astype(bool)
        ground_truth[:5, :] = False
        ground_truth_soft = np.ones(shape=(10, 10)).astype(bool)
        ground_truth_soft[:4, :] = False
        preds = np.ones(shape=(10, 10)).astype(bool)
        ground_truth_soft[:5, :] = False

        rec = recall(ground_truth=ground_truth, preds=preds, ground_truth_soft=ground_truth_soft)
        assert rec == 1.0

        preds[:4, :] = True
        rec = recall(ground_truth=ground_truth, preds=preds, ground_truth_soft=ground_truth_soft)
        assert rec == 1.0

    def test_recallatk(self):
        similarity = np.random.rand(10, 10)
        ground_truth = similarity >= 0.5
        ratk = recallatk(ground_truth=ground_truth, similarity=similarity, k=1)
        assert ratk == 1.0

        ratk = recallatk(ground_truth=ground_truth, similarity=similarity, k=3)
        assert ratk == 1.0

        ratk = recallatk(ground_truth=ground_truth, similarity=similarity, k=5)
        assert ratk == 1.0
