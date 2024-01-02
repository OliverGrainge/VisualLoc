import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms


def test_gt(inriaholidays):
    assert len(inriaholidays.ground_truth()) == len(inriaholidays.query_paths)


def test_query_load(inriaholidays):
    img = Image.open(inriaholidays.query_paths[0])
    assert isinstance(img, Image.Image)


def test_map_load(inriaholidays):
    img = Image.open(inriaholidays.map_paths[0])
    assert isinstance(img, Image.Image)


def test_query_basic(inriaholidays):
    loader = inriaholidays.query_images_loader()
    for idx, batch in loader:
        assert isinstance(batch, torch.Tensor)
        assert batch.dtype == torch.uint8
        break


def test_map_basic(inriaholidays):
    loader = inriaholidays.query_images_loader()
    for idx, batch in loader:
        assert isinstance(batch, torch.Tensor)
        assert batch.dtype == torch.uint8
        break


def test_query_batch_size(inriaholidays):
    loader = inriaholidays.query_images_loader(batch_size=1)
    for idx, batch in loader:
        assert batch.shape[0] == 1
        break


def test_query_map_size(inriaholidays):
    loader = inriaholidays.query_images_loader(batch_size=1)
    for idx, batch in loader:
        assert batch.shape[0] == 1
        break


def test_query_num_workers(inriaholidays):
    loader = inriaholidays.query_images_loader(batch_size=10, num_workers=4)
    for idx, batch in loader:
        assert batch.shape[0] == 10
        break


def test_query_num_workers(inriaholidays):
    loader = inriaholidays.query_images_loader(batch_size=10, num_workers=4)
    for idx, batch in loader:
        assert batch.shape[0] == 10
        break


def test_query_preprocess(inriaholidays):
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Resize((100, 50), antialias=True)])
    loader = inriaholidays.query_images_loader(preprocess=preprocess)
    for idx, batch in loader:
        assert len(batch.shape) == 4
        assert batch.shape[2] == 100
        assert batch.shape[3] == 50
        break


def test_map_preprocess(inriaholidays):
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Resize((100, 50), antialias=True)])
    loader = inriaholidays.map_images_loader(preprocess=preprocess)
    for idx, batch in loader:
        assert len(batch.shape) == 4
        assert batch.shape[2] == 100
        assert batch.shape[3] == 50
        break


def test_gt_maxrange(inriaholidays):
    max_value = max([array.max() for array in inriaholidays.ground_truth()])
    assert max_value <= len(inriaholidays.map_paths)


def test_gt_minrange(inriaholidays):
    min_value = min([array.min() for array in inriaholidays.ground_truth()])
    assert min_value == 0


def test_gt_minrange(inriaholidays):
    assert inriaholidays.ground_truth()[0].dtype == int


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Full training tests require a GPU")
def test_query_pin_memory(inriaholidays):
    loader = inriaholidays.query_images_loader(batch_size=10, pin_memory=True)
    for idx, batch in loader:
        assert batch.shape[0] == 10
        break


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Full training tests require a GPU")
def test_map_pin_memory(inriaholidays):
    loader = inriaholidays.map_images_loader(batch_size=10, pin_memory=True)
    for idx, batch in loader:
        assert batch.shape[0] == 10
        break
