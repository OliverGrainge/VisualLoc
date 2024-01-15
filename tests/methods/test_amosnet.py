import os

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from PlaceRec import Methods
from PlaceRec.utils import get_config

config = get_config()


def test_name(amosnet):
    assert isinstance(amosnet.name, str)
    assert amosnet.name.islower()


def test_device(amosnet):
    assert amosnet.device in ["cpu", "cuda", "mps"]


def test_feature_dim(amosnet):
    assert isinstance(amosnet.features_dim, int)


def test_preprocess(amosnet):
    assert isinstance(amosnet.preprocess, transforms.Compose)


def test_set_query(amosnet):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    amosnet.set_query(test_descriptors)
    assert (amosnet.query_desc == test_descriptors).all()


def test_set_map(amosnet):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    amosnet.set_map(test_descriptors)
    assert (amosnet.map_desc == test_descriptors).all()


def test_set_descriptors(amosnet):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    amosnet.set_query(test_descriptors)
    amosnet.save_descriptors("tmp")


def test_save_descriptors(amosnet):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    amosnet.set_query(test_descriptors)
    test_descriptors = np.random.rand(20, 65).astype(np.float32)
    amosnet.set_map(test_descriptors)
    amosnet.save_descriptors("tmp")


def test_load_descriptors(amosnet):
    query_desc = np.random.rand(10, 65).astype(np.float32)
    amosnet.set_query(query_desc)
    map_desc = np.random.rand(20, 65).astype(np.float32)
    amosnet.set_map(map_desc)
    amosnet.save_descriptors("tmp")
    amosnet.load_descriptors("tmp")
    assert (query_desc == amosnet.query_desc).all()
    assert (map_desc == amosnet.map_desc).all()


def test_compute_query_desc(dataset, amosnet):
    dataloader = dataset.query_images_loader(
        batch_size=1, preprocess=amosnet.preprocess
    )
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = amosnet.compute_query_desc(dataloader=dataloader, pbar=False)
    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float32
    assert np.isclose(np.linalg.norm(desc, axis=1), 1, atol=0.0001).all()
    assert desc.shape[1] == amosnet.features_dim


def test_compute_map_desc(dataset, amosnet):
    dataloader = dataset.map_images_loader(batch_size=1, preprocess=amosnet.preprocess)
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = amosnet.compute_map_desc(dataloader=dataloader, pbar=False)
    print(desc.shape, amosnet.features_dim)
    print(np.linalg.norm(desc, axis=1), np.ones(2))
    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float32
    assert np.isclose(np.linalg.norm(desc, axis=1), 1, atol=0.0001).all()
    assert desc.shape[1] == amosnet.features_dim


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Full training tests require a Nvidia GPU"
)
def test_cuda_acceleration(dataset, amosnet):
    amosnet.set_device("cuda")
    dataloader = dataset.map_images_loader(batch_size=1, preprocess=amosnet.preprocess)
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = amosnet.compute_map_desc(dataloader=dataloader, pbar=False)


@pytest.mark.skipif(
    not os.path.exists(config["weights_directory"]),
    reason="Full training tests require downloaded weights",
)
def test_loading_weights():
    obj = Methods.AmosNet(pretrained=True)
