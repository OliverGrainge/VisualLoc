import argparse
import unittest

import torch

from PlaceRec.Datasets import GardensPointWalking
from PlaceRec.Methods import CALC, NetVLAD
from PlaceRec.utils import get_method

# List all method modules you want to test


methods_to_test = ["amosnet", "hybridnet", "cosplace", "hog", "netvlad", "calc", "convap"]


def create_test_case(method_module):
    class CustomTestMethod(unittest.TestCase):
        def setUp(self):
            self.method = method_module
            self.dataset = GardensPointWalking()

        def test_name_exists(self):
            assert self.method.name is not None, "need to have a method name"

        def test_name_str(self):
            assert isinstance(self.method.name, str), "Method name must be a string"

        def test_name_lower(self):
            assert self.method.name.islower(), "Method name must be lower case"

        def test_preprocess(self):
            dataloader = self.dataset.query_images_loader(partition="test", preprocess=self.method.preprocess)
            for batch in dataloader:
                isinstance(batch, torch.Tensor), "preprocess must return a torch tensor"
                assert batch.dtype == torch.float32, "preprocess must return datatype of float32"

        def test_query(self):
            query_loader = self.dataset.query_images_loader("test", preprocess=self.method.preprocess)
            query_desc = self.method.compute_query_desc(dataloader=query_loader, pbar=False)
            self.assertIn("query_descriptors", query_desc)
            assert self.method.query_desc is not None

        def test_map(self):
            map_loader = self.dataset.query_images_loader("test", preprocess=self.method.preprocess)
            map_desc = self.method.compute_map_desc(dataloader=map_loader, pbar=False)
            self.assertIn("map_descriptors", map_desc)
            assert self.method.map_desc is not None

        def test_descriptors(self):
            query_loader = self.dataset.query_images_loader("test", preprocess=self.method.preprocess)
            self.method.compute_query_desc(dataloader=query_loader, pbar=False)
            del query_loader
            map_loader = self.dataset.query_images_loader("test", preprocess=self.method.preprocess)
            self.method.compute_map_desc(dataloader=map_loader, pbar=False)
            del map_loader
            q_desc = self.method.query_desc
            m_desc = self.method.map_desc
            self.method.save_descriptors("testing_dataset")
            self.method.query_desc = 0
            self.method.map_desc = 0
            self.method.load_descriptors("testing_dataset")
            new_q_desc = self.method.query_desc
            new_m_desc = self.method.map_desc
            assert (q_desc["query_descriptors"] == new_q_desc["query_descriptors"]).all()
            assert (m_desc["map_descriptors"] == new_m_desc["map_descriptors"]).all()

    # Rename the test class to include the method module's name for clarity
    CustomTestMethod.__name__ = f"Test{method_module.name}"

    return CustomTestMethod


# Generate test classes dynamically and set them in the global namespace
for method_name in methods_to_test:
    method = get_method(method_name)
    test_class = create_test_case(method)
    globals()[test_class.__name__] = test_class


if __name__ == "__main__":
    unittest.main()
