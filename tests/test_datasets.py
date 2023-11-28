import argparse
import unittest

import numpy as np
import torch
from torchvision import transforms

from PlaceRec.Datasets import GardensPointWalking
from PlaceRec.utils import get_dataset

# List all method modules you want to test


datasets_to_test = ["pitts30k"]


def create_test_case(dataset_module):
    class CustomTestMethod(unittest.TestCase):
        def setUp(self):
            self.ds = dataset_module

        def test_verify_both_setups_run(self):
            assert isinstance(self.ds.name, str)
            assert self.ds.name.islower()

        def test_query_sequence(self):
            loader = self.ds.query_images_loader("all", shuffle=False)
            for batch in loader:
                batch1 = batch
                break
            loader = self.ds.query_images_loader("all", shuffle=False)
            for batch in loader:
                batch2 = batch
                break

            assert (batch1 == batch2).all()
            assert batch1.dtype == torch.uint8

        def test_map_sequence(self):
            loader = self.ds.map_images_loader(partition="train", shuffle=False)
            for batch in loader:
                batch1 = batch
                break
            loader = self.ds.map_images_loader(partition="train", shuffle=False)
            for batch in loader:
                batch2 = batch
                break

            assert (batch1 == batch2).all()
            assert batch1.dtype == torch.uint8

        def test_query_augmentations(self):
            aug = transforms.Compose(
                [
                    transforms.CenterCrop(10),
                    transforms.ToTensor(),
                    transforms.Resize((480, 752), antialias=True),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

            loader = self.ds.query_images_loader("all", preprocess=aug)
            for batch in loader:
                batch1 = batch
                assert isinstance(batch, torch.Tensor)
                break

            aug = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((480, 752), antialias=True),
                ]
            )

            loader = self.ds.query_images_loader("all", preprocess=aug)
            for batch in loader:
                batch2 = batch
                break

            assert not (batch1 == batch2).all()

        def test_query_shuffle(self):
            loader = self.ds.query_images_loader("all", shuffle=True)
            for batch in loader:
                batch1 = batch
                break

            loader = self.ds.query_images_loader("all", shuffle=True)
            for batch in loader:
                batch2 = batch
                break

            assert not (batch1 == batch2).all()

        def test_gt_train(self):
            gt = self.ds.ground_truth(partition="train")
            Qn = len(self.ds.query_images("train"))
            assert len(gt) == Qn

        def test_gt_val(self):
            gt = self.ds.ground_truth(partition="val")
            Qn = len(self.ds.query_images("val"))
            assert len(gt) == Qn

        def test_gt_test(self):
            gt = self.ds.ground_truth(partition="test")
            Qn = len(self.ds.query_images("test"))
            assert len(gt) == Qn

        def test_gt_all(self):
            gt = self.ds.ground_truth(partition="all")
            Qn = len(self.ds.query_images("all"))
            assert len(gt) == Qn

        def test_gt_size0(self):
            gt = self.ds.ground_truth(partition="train")
            assert len(gt) == self.ds.query_images_loader("train").dataset.__len__()

        def test_gt_size1(self):
            gt = self.ds.ground_truth(partition="test")
            assert len(gt) == len(self.ds.query_images("test"))
            assert len(gt) == self.ds.query_images_loader("test").dataset.__len__()

        def test_gt_size2(self):
            gt = self.ds.ground_truth(partition="val")
            assert len(gt) == len(self.ds.query_images("val"))
            assert len(gt) == self.ds.query_images_loader("val").dataset.__len__()

        def test_gt_size3(self):
            gt = self.ds.ground_truth(partition="all")
            assert len(gt) == len(self.ds.query_images("all"))
            assert len(gt) == self.ds.query_images_loader("all").dataset.__len__()

    # Rename the test class to include the method module's name for clarity
    CustomTestMethod.__name__ = f"Test{dataset_module.name}"

    return CustomTestMethod


# Generate test classes dynamically and set them in the global namespace
for dataset_name in datasets_to_test:
    dataset = get_dataset(dataset_name)
    test_class = create_test_case(dataset)
    globals()[test_class.__name__] = test_class


if __name__ == "__main__":
    unittest.main()
