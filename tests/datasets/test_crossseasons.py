import pytest
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

def test_paths(crossseasons):
    assert len(crossseasons.query_paths) == len(crossseasons.map_paths)

def test_gt(crossseasons):
    assert len(crossseasons.ground_truth()) == len(crossseasons.query_paths)

def test_query_load(crossseasons):
    img = Image.open(crossseasons.query_paths[0])
    assert isinstance(img, Image.Image)

def test_map_load(crossseasons):
    img = Image.open(crossseasons.map_paths[0])
    assert isinstance(img, Image.Image)

def test_query_basic(crossseasons):
    loader = crossseasons.query_images_loader()
    for idx, batch in loader:
        assert isinstance(batch, torch.Tensor)
        assert batch.dtype == torch.uint8
        break

def test_map_basic(crossseasons):
    loader = crossseasons.query_images_loader()
    for idx, batch in loader:
        assert isinstance(batch, torch.Tensor)
        assert batch.dtype == torch.uint8
        break

def test_query_batch_size(crossseasons):
    loader = crossseasons.query_images_loader(batch_size=1)
    for idx, batch in loader:
        assert batch.shape[0] == 1
        break

def test_query_map_size(crossseasons):
    loader = crossseasons.query_images_loader(batch_size=1)
    for idx, batch in loader:
        assert batch.shape[0] == 1
        break

def test_query_num_workers(crossseasons):
    loader = crossseasons.query_images_loader(batch_size=10, num_workers=4)
    for idx, batch in loader:
        assert batch.shape[0] == 10
        break

def test_query_num_workers(crossseasons):
    loader = crossseasons.query_images_loader(batch_size=10, num_workers=4)
    for idx, batch in loader:
        assert batch.shape[0] == 10
        break

def test_query_preprocess(crossseasons):
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Resize((100, 50), antialias=True)])
    loader = crossseasons.query_images_loader(preprocess=preprocess)
    for idx, batch in loader:
        assert len(batch.shape) == 4
        assert batch.shape[2] == 100
        assert batch.shape[3] == 50
        break

def test_map_preprocess(crossseasons):
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Resize((100, 50), antialias=True)])
    loader = crossseasons.map_images_loader(preprocess=preprocess)
    for idx, batch in loader:
        assert len(batch.shape) == 4
        assert batch.shape[2] == 100
        assert batch.shape[3] == 50
        break

def test_gt_maxrange(crossseasons):
    max_value = max([array.max() for array in crossseasons.ground_truth()])
    assert max_value <= len(crossseasons.map_paths)

def test_gt_minrange(crossseasons):
    min_value = min([array.min() for array in crossseasons.ground_truth()])
    assert min_value == 0

def test_gt_minrange(crossseasons):
    assert crossseasons.ground_truth()[0].dtype == int

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Full training tests require a GPU"
)
def test_query_pin_memory(crossseasons):
    loader = crossseasons.query_images_loader(batch_size=10, pin_memory=True)
    for idx, batch in loader:
        assert batch.shape[0] == 10
        break

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Full training tests require a GPU"
)
def test_map_pin_memory(crossseasons):
    loader = crossseasons.map_images_loader(batch_size=10, pin_memory=True)
    for idx, batch in loader:
        assert batch.shape[0] == 10
        break





