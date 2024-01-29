import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms


def test_gt(mapillarysls):
    assert len(mapillarysls.ground_truth()) == len(mapillarysls.query_paths)


def test_query_load(mapillarysls):
    img = Image.open(mapillarysls.query_paths[0])
    assert isinstance(img, Image.Image)


def test_map_load(mapillarysls):
    img = Image.open(mapillarysls.map_paths[0])
    assert isinstance(img, Image.Image)


def test_query_basic(mapillarysls):
    loader = mapillarysls.query_images_loader()
    for idx, batch in loader:
        assert isinstance(batch, torch.Tensor)
        assert batch.dtype == torch.uint8
        break


def test_map_basic(mapillarysls):
    loader = mapillarysls.query_images_loader()
    for idx, batch in loader:
        assert isinstance(batch, torch.Tensor)
        assert batch.dtype == torch.uint8
        break


def test_query_batch_size(mapillarysls):
    loader = mapillarysls.query_images_loader(batch_size=1)
    for idx, batch in loader:
        assert batch.shape[0] == 1
        break


def test_query_map_size(mapillarysls):
    loader = mapillarysls.query_images_loader(batch_size=1)
    for idx, batch in loader:
        assert batch.shape[0] == 1
        break


def test_query_num_workers(mapillarysls):
    loader = mapillarysls.query_images_loader(batch_size=10, num_workers=4)
    for idx, batch in loader:
        assert batch.shape[0] == 10
        break


def test_query_num_workers(mapillarysls):
    loader = mapillarysls.query_images_loader(batch_size=10, num_workers=4)
    for idx, batch in loader:
        assert batch.shape[0] == 10
        break


def test_query_preprocess(mapillarysls):
    preprocess = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((100, 50), antialias=True)]
    )
    loader = mapillarysls.query_images_loader(preprocess=preprocess)
    for idx, batch in loader:
        assert len(batch.shape) == 4
        assert batch.shape[2] == 100
        assert batch.shape[3] == 50
        break


def test_map_preprocess(mapillarysls):
    preprocess = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((100, 50), antialias=True)]
    )
    loader = mapillarysls.map_images_loader(preprocess=preprocess)
    for idx, batch in loader:
        assert len(batch.shape) == 4
        assert batch.shape[2] == 100
        assert batch.shape[3] == 50
        break


def test_gt_maxrange(mapillarysls):
    max_value = max([array.max() for array in mapillarysls.ground_truth()])
    assert max_value <= len(mapillarysls.map_paths)


def test_gt_minrange(mapillarysls):
    min_value = min([array.min() for array in mapillarysls.ground_truth()])
    assert min_value == 0


def test_gt_minrange(mapillarysls):
    assert mapillarysls.ground_truth()[0].dtype == int


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Full training tests require a GPU"
)
def test_query_pin_memory(mapillarysls):
    loader = mapillarysls.query_images_loader(batch_size=10, pin_memory=True)
    for idx, batch in loader:
        assert batch.shape[0] == 10
        break


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Full training tests require a GPU"
)
def test_map_pin_memory(mapillarysls):
    loader = mapillarysls.map_images_loader(batch_size=10, pin_memory=True)
    for idx, batch in loader:
        assert batch.shape[0] == 10
        break
