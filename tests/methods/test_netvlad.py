import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from PlaceRec import Methods
from PlaceRec.utils import get_config 
import os

config = get_config()


def test_name(netvlad):
    assert isinstance(netvlad.name, str)
    assert netvlad.name.islower()


def test_device(netvlad):
    assert netvlad.device in ["cpu", "cuda", "mps"]


def test_feature_dim(netvlad):
    assert isinstance(netvlad.features_dim, int)


def test_preprocess(netvlad):
    assert isinstance(netvlad.preprocess, transforms.Compose)


def test_set_query(netvlad):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    netvlad.set_query(test_descriptors)
    assert (netvlad.query_desc == test_descriptors).all()


def test_set_map(netvlad):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    netvlad.set_map(test_descriptors)
    assert (netvlad.map_desc == test_descriptors).all()


def test_set_descriptors(netvlad):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    netvlad.set_query(test_descriptors)
    netvlad.save_descriptors("tmp")


def test_save_descriptors(netvlad):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    netvlad.set_query(test_descriptors)
    test_descriptors = np.random.rand(20, 65).astype(np.float32)
    netvlad.set_map(test_descriptors)
    netvlad.save_descriptors("tmp")


def test_load_descriptors(netvlad):
    query_desc = np.random.rand(10, 65).astype(np.float32)
    netvlad.set_query(query_desc)
    map_desc = np.random.rand(20, 65).astype(np.float32)
    netvlad.set_map(map_desc)
    netvlad.save_descriptors("tmp")
    netvlad.load_descriptors("tmp")
    assert (query_desc == netvlad.query_desc).all()
    assert (map_desc == netvlad.map_desc).all()


def test_compute_query_desc(dataset, netvlad):
    dataloader = dataset.query_images_loader(batch_size=1, preprocess=netvlad.preprocess)
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = netvlad.compute_query_desc(dataloader=dataloader, pbar=False)
    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float32
    assert np.isclose(np.linalg.norm(desc, axis=1), 1, atol=0.0001).all()
    assert desc.shape[1] == netvlad.features_dim


def test_compute_map_desc(dataset, netvlad):
    dataloader = dataset.map_images_loader(batch_size=1, preprocess=netvlad.preprocess)
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = netvlad.compute_map_desc(dataloader=dataloader, pbar=False)
    print(desc.shape, netvlad.features_dim)
    print(np.linalg.norm(desc, axis=1), np.ones(2))
    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float32
    assert np.isclose(np.linalg.norm(desc, axis=1), 1, atol=0.0001).all()
    assert desc.shape[1] == netvlad.features_dim


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Full training tests require a Nvidia GPU")
def test_cuda_acceleration(dataset, netvlad):
    netvlad.set_device("cuda")
    dataloader = dataset.map_images_loader(batch_size=1, preprocess=netvlad.preprocess)
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = netvlad.compute_map_desc(dataloader=dataloader, pbar=False)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Full training tests require a apple GPU")
def test_mps_acceleration(dataset, netvlad):
    netvlad.set_device("mps")
    dataloader = dataset.map_images_loader(batch_size=1, preprocess=netvlad.preprocess)
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = netvlad.compute_map_desc(dataloader=dataloader, pbar=False)


@pytest.mark.skipif(not os.path.exists(config["weights_directory"]), reason="Full training tests require downloaded weights")
def test_loading_weights():
    obj = Methods.NetVLAD(pretrained=True)