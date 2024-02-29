import os

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from PlaceRec import Methods
from PlaceRec.utils import get_config

config = get_config()


def test_name(eigenplaces):
    assert isinstance(eigenplaces.name, str)
    assert eigenplaces.name.islower()


def test_device(eigenplaces):
    assert eigenplaces.device in ["cpu", "cuda", "mps"]


def test_feature_dim(eigenplaces):
    assert isinstance(eigenplaces.features_dim, int)


def test_preprocess(eigenplaces):
    assert isinstance(eigenplaces.preprocess, transforms.Compose)


def test_set_query(eigenplaces):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    eigenplaces.set_query(test_descriptors)
    assert (eigenplaces.query_desc == test_descriptors).all()


def test_set_map(eigenplaces):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    eigenplaces.set_map(test_descriptors)
    assert (eigenplaces.map_desc == test_descriptors).all()


def test_set_descriptors(eigenplaces):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    eigenplaces.set_query(test_descriptors)
    eigenplaces.save_descriptors("tmp")


def test_save_descriptors(eigenplaces):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    eigenplaces.set_query(test_descriptors)
    test_descriptors = np.random.rand(20, 65).astype(np.float32)
    eigenplaces.set_map(test_descriptors)
    eigenplaces.save_descriptors("tmp")


def test_load_descriptors(eigenplaces):
    query_desc = np.random.rand(10, 65).astype(np.float32)
    eigenplaces.set_query(query_desc)
    map_desc = np.random.rand(20, 65).astype(np.float32)
    eigenplaces.set_map(map_desc)
    eigenplaces.save_descriptors("tmp")
    eigenplaces.load_descriptors("tmp")
    assert (query_desc == eigenplaces.query_desc).all()
    assert (map_desc == eigenplaces.map_desc).all()


def test_compute_query_desc(dataset, eigenplaces):
    dataloader = dataset.query_images_loader(
        batch_size=1, preprocess=eigenplaces.preprocess
    )
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = eigenplaces.compute_query_desc(dataloader=dataloader, pbar=False)
    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float32
    assert np.isclose(np.linalg.norm(desc, axis=1), 1, atol=0.0001).all()
    assert desc.shape[1] == eigenplaces.features_dim


def test_compute_map_desc(dataset, eigenplaces):
    dataloader = dataset.map_images_loader(
        batch_size=1, preprocess=eigenplaces.preprocess
    )
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = eigenplaces.compute_map_desc(dataloader=dataloader, pbar=False)
    print(desc.shape, eigenplaces.features_dim)
    print(np.linalg.norm(desc, axis=1), np.ones(2))
    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float32
    assert np.isclose(np.linalg.norm(desc, axis=1), 1, atol=0.0001).all()
    assert desc.shape[1] == eigenplaces.features_dim


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Full training tests require a Nvidia GPU"
)
def test_cuda_acceleration(dataset, eigenplaces):
    eigenplaces.set_device("cuda")
    dataloader = dataset.map_images_loader(
        batch_size=1, preprocess=eigenplaces.preprocess
    )
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = eigenplaces.compute_map_desc(dataloader=dataloader, pbar=False)


@pytest.mark.skipif(
    not os.path.exists(config["weights_directory"]),
    reason="Full training tests require downloaded weights",
)
def test_loading_weights():
    obj = Methods.ResNet50_EigenPlaces(pretrained=True)
