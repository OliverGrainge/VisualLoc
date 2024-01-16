import os

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from PlaceRec import Methods
from PlaceRec.utils import get_config

config = get_config()


def test_name(resnet50_gem):
    assert isinstance(resnet50_gem.name, str)
    assert resnet50_gem.name.islower()


def test_device(resnet50_gem):
    assert resnet50_gem.device in ["cpu", "cuda", "mps"]


def test_feature_dim(resnet50_gem):
    assert isinstance(resnet50_gem.features_dim, int)


def test_preprocess(resnet50_gem):
    assert isinstance(resnet50_gem.preprocess, transforms.Compose)


def test_set_query(resnet50_gem):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    resnet50_gem.set_query(test_descriptors)
    assert (resnet50_gem.query_desc == test_descriptors).all()


def test_set_map(resnet50_gem):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    resnet50_gem.set_map(test_descriptors)
    assert (resnet50_gem.map_desc == test_descriptors).all()


def test_set_descriptors(resnet50_gem):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    resnet50_gem.set_query(test_descriptors)
    resnet50_gem.save_descriptors("tmp")


def test_save_descriptors(resnet50_gem):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    resnet50_gem.set_query(test_descriptors)
    test_descriptors = np.random.rand(20, 65).astype(np.float32)
    resnet50_gem.set_map(test_descriptors)
    resnet50_gem.save_descriptors("tmp")


def test_load_descriptors(resnet50_gem):
    query_desc = np.random.rand(10, 65).astype(np.float32)
    resnet50_gem.set_query(query_desc)
    map_desc = np.random.rand(20, 65).astype(np.float32)
    resnet50_gem.set_map(map_desc)
    resnet50_gem.save_descriptors("tmp")
    resnet50_gem.load_descriptors("tmp")
    assert (query_desc == resnet50_gem.query_desc).all()
    assert (map_desc == resnet50_gem.map_desc).all()


def test_compute_query_desc(dataset, resnet50_gem):
    dataloader = dataset.query_images_loader(
        batch_size=1, preprocess=resnet50_gem.preprocess
    )
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = resnet50_gem.compute_query_desc(dataloader=dataloader, pbar=False)
    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float32
    assert np.isclose(np.linalg.norm(desc, axis=1), 1, atol=0.0001).all()
    assert desc.shape[1] == resnet50_gem.features_dim


def test_compute_map_desc(dataset, resnet50_gem):
    dataloader = dataset.map_images_loader(
        batch_size=1, preprocess=resnet50_gem.preprocess
    )
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = resnet50_gem.compute_map_desc(dataloader=dataloader, pbar=False)
    print(desc.shape, resnet50_gem.features_dim)
    print(np.linalg.norm(desc, axis=1), np.ones(2))
    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float32
    assert np.isclose(np.linalg.norm(desc, axis=1), 1, atol=0.0001).all()
    assert desc.shape[1] == resnet50_gem.features_dim


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Full training tests require a Nvidia GPU"
)
def test_cuda_acceleration(dataset, resnet50_gem):
    resnet50_gem.set_device("cuda")
    dataloader = dataset.map_images_loader(
        batch_size=1, preprocess=resnet50_gem.preprocess
    )
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = resnet50_gem.compute_map_desc(dataloader=dataloader, pbar=False)



@pytest.mark.skipif(
    not os.path.exists(config["weights_directory"]),
    reason="Full training tests require downloaded weights",
)
def test_loading_weights():
    obj = Methods.ResNet50GeM(pretrained=True)
