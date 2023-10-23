import sys

sys.path.append("/Users/olivergrainge/Documents/github/VisualLoc")

import unittest

import numpy as np
from torchvision import transforms

from PlaceRec.Datasets import GardensPointWalking
from PlaceRec.Methods import RegionVLAD


class setup_test(unittest.TestCase):
    def setUp(self):
        self.method = RegionVLAD()
        self.ds = GardensPointWalking()
        self.sample_size = 3


class GenericMethodTest(setup_test):
    def test_name(self):
        assert isinstance(self.method.name, str)
        assert self.method.name.islower()

    def test_query_desc(self):
        Q = self.ds.query_images("test", preprocess=self.method.preprocess)[: self.sample_size]
        res = self.method.compute_query_desc(images=Q, pbar=False)
        assert isinstance(res, dict)

    def test_map_desc(self):
        M = self.ds.map_images(partition="test", preprocess=self.method.preprocess)[: self.sample_size]
        res = self.method.compute_map_desc(images=M, pbar=False)
        assert isinstance(res, dict)

    def test_similarity_matrix(self):
        query_images = self.ds.query_images(partition="test", preprocess=self.method.preprocess)[: self.sample_size]
        map_images = self.ds.map_images("test", preprocess=self.method.preprocess)[: self.sample_size]
        query_desc = self.method.compute_query_desc(images=query_images)
        map_desc = self.method.compute_map_desc(images=map_images)
        S = self.method.similarity_matrix(query_desc, map_desc)
        assert S.shape[0] == map_images.shape[0]
        assert S.shape[1] == query_images.shape[0]
        assert isinstance(S, np.ndarray)
        assert S.dtype == np.float32

    def test_set_map(self):
        map_images = self.ds.map_images(partition="test", preprocess=self.method.preprocess)[: self.sample_size]
        map_desc = self.method.compute_map_desc(images=map_images)
        self.method.set_map(map_desc)
        assert self.method.map is not None

    def test_place_recognise(self):
        query_images = self.ds.query_images("test", preprocess=self.method.preprocess)[: self.sample_size]
        map_images = self.ds.map_images(partition="test", preprocess=self.method.preprocess)[: self.sample_size]

        map_desc = self.method.compute_map_desc(images=map_images)
        self.method.set_map(map_desc)
        idx, score = self.method.place_recognise(images=query_images, top_n=3)
        assert isinstance(idx, np.ndarray)
        assert isinstance(score, np.ndarray)
        assert idx.shape[0] == query_images.shape[0]
        assert idx.shape[1] == 3
        assert score.shape[0] == query_images.shape[0]
        assert score.shape[1] == 3
        assert idx.dtype == int
        assert score.dtype == np.float32

    def test_save_and_load(self):
        query_images = self.ds.query_images("test", preprocess=self.method.preprocess)[: self.sample_size]
        map_images = self.ds.map_images(partition="test", preprocess=self.method.preprocess)[: self.sample_size]
        map_desc = self.method.compute_map_desc(images=map_images)
        query_desc = self.method.compute_query_desc(images=query_images)
        self.method.set_map(map_desc)
        self.method.set_query(query_desc)

        self.method.save_descriptors(self.ds.name)
        self.method.load_descriptors(self.ds.name)


if __name__ == "__main__":
    unittest.main()
