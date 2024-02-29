import os

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from PlaceRec import Methods
from PlaceRec.utils import get_config

config = get_config()


def test_name(hybridnet):
    assert isinstance(hybridnet.name, str)
    assert hybridnet.name.islower()


def test_device(hybridnet):
    assert hybridnet.device in ["cpu", "cuda", "mps"]


def test_feature_dim(hybridnet):
    assert isinstance(hybridnet.features_dim, int)


def test_preprocess(hybridnet):
    assert isinstance(hybridnet.preprocess, transforms.Compose)


def test_set_query(hybridnet):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    hybridnet.set_query(test_descriptors)
    assert (hybridnet.query_desc == test_descriptors).all()


def test_set_map(hybridnet):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    hybridnet.set_map(test_descriptors)
    assert (hybridnet.map_desc == test_descriptors).all()


def test_set_descriptors(hybridnet):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    hybridnet.set_query(test_descriptors)
    hybridnet.save_descriptors("tmp")


def test_save_descriptors(hybridnet):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    hybridnet.set_query(test_descriptors)
    test_descriptors = np.random.rand(20, 65).astype(np.float32)
    hybridnet.set_map(test_descriptors)
    hybridnet.save_descriptors("tmp")


def test_load_descriptors(hybridnet):
    query_desc = np.random.rand(10, 65).astype(np.float32)
    hybridnet.set_query(query_desc)
    map_desc = np.random.rand(20, 65).astype(np.float32)
    hybridnet.set_map(map_desc)
    hybridnet.save_descriptors("tmp")
    hybridnet.load_descriptors("tmp")
    assert (query_desc == hybridnet.query_desc).all()
    assert (map_desc == hybridnet.map_desc).all()


def test_compute_query_desc(dataset, hybridnet):
    dataloader = dataset.query_images_loader(
        batch_size=1, preprocess=hybridnet.preprocess
    )
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = hybridnet.compute_query_desc(dataloader=dataloader, pbar=False)
    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float32
    assert np.isclose(np.linalg.norm(desc, axis=1), 1, atol=0.0001).all()
    assert desc.shape[1] == hybridnet.features_dim


def test_compute_map_desc(dataset, hybridnet):
    dataloader = dataset.map_images_loader(
        batch_size=1, preprocess=hybridnet.preprocess
    )
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = hybridnet.compute_map_desc(dataloader=dataloader, pbar=False)
    print(desc.shape, hybridnet.features_dim)
    print(np.linalg.norm(desc, axis=1), np.ones(2))
    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float32
    assert np.isclose(np.linalg.norm(desc, axis=1), 1, atol=0.0001).all()
    assert desc.shape[1] == hybridnet.features_dim


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Full training tests require a Nvidia GPU"
)
def test_cuda_acceleration(dataset, hybridnet):
    hybridnet.set_device("cuda")
    dataloader = dataset.map_images_loader(
        batch_size=1, preprocess=hybridnet.preprocess
    )
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = hybridnet.compute_map_desc(dataloader=dataloader, pbar=False)


@pytest.mark.skipif(
    not os.path.exists(config["weights_directory"]),
    reason="Full training tests require downloaded weights",
)
def test_loading_weights():
    obj = Methods.AlexNet_HybridNet(pretrained=True)
