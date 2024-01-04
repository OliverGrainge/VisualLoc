import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms


def test_gt(pitts30k):
    assert len(pitts30k.ground_truth()) == len(pitts30k.query_paths)


def test_query_load(pitts30k):
    img = Image.open(pitts30k.query_paths[0])
    assert isinstance(img, Image.Image)


def test_map_load(pitts30k):
    img = Image.open(pitts30k.map_paths[0])
    assert isinstance(img, Image.Image)


def test_query_basic(pitts30k):
    loader = pitts30k.query_images_loader()
    for idx, batch in loader:
        assert isinstance(batch, torch.Tensor)
        assert batch.dtype == torch.uint8
        break


def test_map_basic(pitts30k):
    loader = pitts30k.query_images_loader()
    for idx, batch in loader:
        assert isinstance(batch, torch.Tensor)
        assert batch.dtype == torch.uint8
        break


def test_query_batch_size(pitts30k):
    loader = pitts30k.query_images_loader(batch_size=1)
    for idx, batch in loader:
        assert batch.shape[0] == 1
        break


def test_query_map_size(pitts30k):
    loader = pitts30k.query_images_loader(batch_size=1)
    for idx, batch in loader:
        assert batch.shape[0] == 1
        break


def test_query_num_workers(pitts30k):
    loader = pitts30k.query_images_loader(batch_size=10, num_workers=4)
    for idx, batch in loader:
        assert batch.shape[0] == 10
        break


def test_query_num_workers(pitts30k):
    loader = pitts30k.query_images_loader(batch_size=10, num_workers=4)
    for idx, batch in loader:
        assert batch.shape[0] == 10
        break


def test_query_preprocess(pitts30k):
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Resize((100, 50), antialias=True)])
    loader = pitts30k.query_images_loader(preprocess=preprocess)
    for idx, batch in loader:
        assert len(batch.shape) == 4
        assert batch.shape[2] == 100
        assert batch.shape[3] == 50
        break


def test_map_preprocess(pitts30k):
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Resize((100, 50), antialias=True)])
    loader = pitts30k.map_images_loader(preprocess=preprocess)
    for idx, batch in loader:
        assert len(batch.shape) == 4
        assert batch.shape[2] == 100
        assert batch.shape[3] == 50
        break


def test_gt_maxrange(pitts30k):
    max_value = max([array.max() for array in pitts30k.ground_truth()])
    assert max_value <= len(pitts30k.map_paths)


def test_gt_minrange(pitts30k):
    min_value = min([array.min() for array in pitts30k.ground_truth()])
    assert min_value == 0


def test_gt_minrange(pitts30k):
    assert pitts30k.ground_truth()[0].dtype == int


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Full training tests require a GPU")
def test_query_pin_memory(pitts30k):
    loader = pitts30k.query_images_loader(batch_size=10, pin_memory=True)
    for idx, batch in loader:
        assert batch.shape[0] == 10
        break


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Full training tests require a GPU")
def test_map_pin_memory(pitts30k):
    loader = pitts30k.map_images_loader(batch_size=10, pin_memory=True)
    for idx, batch in loader:
        assert batch.shape[0] == 10
        break
