import os

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from PlaceRec import Methods
from PlaceRec.utils import get_config

config = get_config()


def test_name(gatinginceptionlarge):
    assert isinstance(gatinginceptionlarge.name, str)
    assert gatinginceptionlarge.name.islower()


def test_device(gatinginceptionlarge):
    assert gatinginceptionlarge.device in ["cpu", "cuda", "mps"]


def test_feature_dim(gatinginceptionlarge):
    assert isinstance(gatinginceptionlarge.features_dim, int)


def test_preprocess(gatinginceptionlarge):
    assert isinstance(gatinginceptionlarge.preprocess, transforms.Compose)


def test_set_query(gatinginceptionlarge):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    gatinginceptionlarge.set_query(test_descriptors)
    assert (gatinginceptionlarge.query_desc == test_descriptors).all()


def test_set_map(gatinginceptionlarge):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    gatinginceptionlarge.set_map(test_descriptors)
    assert (gatinginceptionlarge.map_desc == test_descriptors).all()


def test_set_descriptors(gatinginceptionlarge):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    gatinginceptionlarge.set_query(test_descriptors)
    gatinginceptionlarge.save_descriptors("tmp")


def test_save_descriptors(gatinginceptionlarge):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    gatinginceptionlarge.set_query(test_descriptors)
    test_descriptors = np.random.rand(20, 65).astype(np.float32)
    gatinginceptionlarge.set_map(test_descriptors)
    gatinginceptionlarge.save_descriptors("tmp")


def test_load_descriptors(gatinginceptionlarge):
    query_desc = np.random.rand(10, 65).astype(np.float32)
    gatinginceptionlarge.set_query(query_desc)
    map_desc = np.random.rand(20, 65).astype(np.float32)
    gatinginceptionlarge.set_map(map_desc)
    gatinginceptionlarge.save_descriptors("tmp")
    gatinginceptionlarge.load_descriptors("tmp")
    assert (query_desc == gatinginceptionlarge.query_desc).all()
    assert (map_desc == gatinginceptionlarge.map_desc).all()


def test_compute_query_desc(dataset, gatinginceptionlarge):
    dataloader = dataset.query_images_loader(batch_size=1, preprocess=gatinginceptionlarge.preprocess)
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = gatinginceptionlarge.compute_query_desc(dataloader=dataloader, pbar=False)
    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float32
    assert np.isclose(np.linalg.norm(desc, axis=1), 1, atol=0.0001).all()
    assert desc.shape[1] == gatinginceptionlarge.features_dim


def test_compute_map_desc(dataset, gatinginceptionlarge):
    dataloader = dataset.map_images_loader(batch_size=1, preprocess=gatinginceptionlarge.preprocess)
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = gatinginceptionlarge.compute_map_desc(dataloader=dataloader, pbar=False)
    print(desc.shape, gatinginceptionlarge.features_dim)
    print(np.linalg.norm(desc, axis=1), np.ones(2))
    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float32
    assert np.isclose(np.linalg.norm(desc, axis=1), 1, atol=0.0001).all()
    assert desc.shape[1] == gatinginceptionlarge.features_dim


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Full training tests require a Nvidia GPU"
)
def test_cuda_acceleration(dataset, gatinginceptionlarge):
    gatinginceptionlarge.set_device("cuda")
    dataloader = dataset.map_images_loader(batch_size=1, preprocess=gatinginceptionlarge.preprocess)
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = gatinginceptionlarge.compute_map_desc(dataloader=dataloader, pbar=False)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Full training tests require a apple GPU",
)
def test_mps_acceleration(dataset, gatinginceptionlarge):
    gatinginceptionlarge.set_device("mps")
    dataloader = dataset.map_images_loader(batch_size=1, preprocess=gatinginceptionlarge.preprocess)
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = gatinginceptionlarge.compute_map_desc(dataloader=dataloader, pbar=False)


@pytest.mark.skipif(
    not os.path.exists(config["weights_directory"]),
    reason="Full training tests require downloaded weights",
)
def test_loading_weights():
    obj = Methods.gatinginceptionlarge(pretrained=True)