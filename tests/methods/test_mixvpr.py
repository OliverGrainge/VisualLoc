import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from PlaceRec import Methods
from PlaceRec.utils import get_config 
import os

config = get_config()


def test_name(mixvpr):
    assert isinstance(mixvpr.name, str)
    assert mixvpr.name.islower()


def test_device(mixvpr):
    assert mixvpr.device in ["cpu", "cuda", "mps"]


def test_feature_dim(mixvpr):
    assert isinstance(mixvpr.features_dim, int)


def test_preprocess(mixvpr):
    assert isinstance(mixvpr.preprocess, transforms.Compose)


def test_set_query(mixvpr):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    mixvpr.set_query(test_descriptors)
    assert (mixvpr.query_desc == test_descriptors).all()


def test_set_map(mixvpr):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    mixvpr.set_map(test_descriptors)
    assert (mixvpr.map_desc == test_descriptors).all()


def test_set_descriptors(mixvpr):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    mixvpr.set_query(test_descriptors)
    mixvpr.save_descriptors("tmp")


def test_save_descriptors(mixvpr):
    test_descriptors = np.random.rand(10, 65).astype(np.float32)
    mixvpr.set_query(test_descriptors)
    test_descriptors = np.random.rand(20, 65).astype(np.float32)
    mixvpr.set_map(test_descriptors)
    mixvpr.save_descriptors("tmp")


def test_load_descriptors(mixvpr):
    query_desc = np.random.rand(10, 65).astype(np.float32)
    mixvpr.set_query(query_desc)
    map_desc = np.random.rand(20, 65).astype(np.float32)
    mixvpr.set_map(map_desc)
    mixvpr.save_descriptors("tmp")
    mixvpr.load_descriptors("tmp")
    assert (query_desc == mixvpr.query_desc).all()
    assert (map_desc == mixvpr.map_desc).all()


def test_compute_query_desc(dataset, mixvpr):
    dataloader = dataset.query_images_loader(batch_size=1, preprocess=mixvpr.preprocess)
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = mixvpr.compute_query_desc(dataloader=dataloader, pbar=False)
    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float32
    assert np.isclose(np.linalg.norm(desc, axis=1), 1, atol=0.0001).all()
    assert desc.shape[1] == mixvpr.features_dim


def test_compute_map_desc(dataset, mixvpr):
    dataloader = dataset.map_images_loader(batch_size=1, preprocess=mixvpr.preprocess)
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = mixvpr.compute_map_desc(dataloader=dataloader, pbar=False)
    print(desc.shape, mixvpr.features_dim)
    print(np.linalg.norm(desc, axis=1), np.ones(2))
    assert isinstance(desc, np.ndarray)
    assert desc.dtype == np.float32
    assert np.isclose(np.linalg.norm(desc, axis=1), 1, atol=0.0001).all()
    assert desc.shape[1] == mixvpr.features_dim


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Full training tests require a Nvidia GPU")
def test_cuda_acceleration(dataset, mixvpr):
    mixvpr.set_device("cuda")
    dataloader = dataset.map_images_loader(batch_size=1, preprocess=mixvpr.preprocess)
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = mixvpr.compute_map_desc(dataloader=dataloader, pbar=False)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Full training tests require a apple GPU")
def test_mps_acceleration(dataset, mixvpr):
    mixvpr.set_device("mps")
    dataloader = dataset.map_images_loader(batch_size=1, preprocess=mixvpr.preprocess)
    dataset = dataloader.dataset
    dataset = Subset(dataset, list(range(2)))
    dataloader = DataLoader(dataset, batch_size=1)
    desc = mixvpr.compute_map_desc(dataloader=dataloader, pbar=False)



@pytest.mark.skipif(not os.path.exists(config["weights_directory"]), reason="Full training tests require downloaded weights")
def test_loading_weights():
    obj = Methods.MixVPR(pretrained=True)