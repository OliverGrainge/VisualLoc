import os

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from PlaceRec import Methods
from PlaceRec.utils import get_config

config = get_config()


def test_name(calc):
    assert isinstance(calc.name, str)
    assert calc.name.islower()


def test_device(calc):
    assert calc.device in ["cpu", "cuda", "mps"]


def test_feature_dim(calc):
    assert isinstance(calc.features_dim, int)


def test_preprocess(calc):
    assert isinstance(calc.preprocess, transforms.Compose)


def test_set_query(calc):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    calc.set_query(test_descriptors)
    assert (calc.query_desc == test_descriptors).all()


def test_set_map(calc):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    calc.set_map(test_descriptors)
    assert (calc.map_desc == test_descriptors).all()


def test_set_descriptors(calc):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    calc.set_query(test_descriptors)
    calc.save_descriptors("tmp")


def test_save_descriptors(calc):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    calc.set_query(test_descriptors)
    test_descriptors = np.random.rand(20, 65).astype(np.float32)
    calc.set_map(test_descriptors)
    calc.save_descriptors("tmp")


def test_load_descriptors(calc):
    query_desc = np.random.rand(10, 65).astype(np.float32)
    calc.set_query(query_desc)
    map_desc = np.random.rand(20, 65).astype(np.float32)
    calc.set_map(map_desc)
    calc.save_descriptors("tmp")
    calc.load_descriptors("tmp")
    assert (query_desc == calc.query_desc).all()
    assert (map_desc == calc.map_desc).all()


def test_compute_query_desc(dataset, calc):
    dataloader = dataset.query_images_loader(batch_size=1, preprocess=calc.preprocess)
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = calc.compute_query_desc(dataloader=dataloader, pbar=False)
    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float32
    assert np.isclose(np.linalg.norm(desc, axis=1), 1, atol=0.0001).all()
    assert desc.shape[1] == calc.features_dim


def test_compute_map_desc(dataset, calc):
    dataloader = dataset.map_images_loader(batch_size=1, preprocess=calc.preprocess)
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = calc.compute_map_desc(dataloader=dataloader, pbar=False)
    print(desc.shape, calc.features_dim)
    print(np.linalg.norm(desc, axis=1), np.ones(2))
    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float32
    assert np.isclose(np.linalg.norm(desc, axis=1), 1, atol=0.0001).all()
    assert desc.shape[1] == calc.features_dim


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Full training tests require a Nvidia GPU"
)
def test_cuda_acceleration(dataset, calc):
    calc.set_device("cuda")
    dataloader = dataset.map_images_loader(batch_size=1, preprocess=calc.preprocess)
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = calc.compute_map_desc(dataloader=dataloader, pbar=False)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Full training tests require a apple GPU",
)
def test_mps_acceleration(dataset, calc):
    calc.set_device("mps")
    dataloader = dataset.map_images_loader(batch_size=1, preprocess=calc.preprocess)
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = calc.compute_map_desc(dataloader=dataloader, pbar=False)


@pytest.mark.skipif(
    not os.path.exists(config["weights_directory"]),
    reason="Full training tests require downloaded weights",
)
def test_loading_weights():
    obj = Methods.CALC(pretrained=True)
