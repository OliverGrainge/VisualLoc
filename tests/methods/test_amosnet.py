import pytest 
import numpy as np
from torch.utils.data import Subset
import torchvision.transforms as transforms

def test_name(amosnet):
    assert isinstance(amosnet.name, str)
    assert amosnet.name.islower()

def test_device(amosnet):
    assert amosnet.device in ["cpu", "cuda", "mps"]

def test_feature_dim(amosnet):
    assert isinstance(amosnet.features_dim, int)

def test_preprocess(amosnet):
    assert isinstance(amosnet, transforms.Compose)

def test_set_query(amosnet):
    desc = np.random.rand(10, 65)
    test_descriptors = {"query_descriptors": desc}
    amosnet.set_query(test_descriptors)
    assert (amosnet.query_desc["query_descriptors"] == desc).all()

def test_set_map(amosnet):
    desc = np.random.rand(10, 65)
    test_descriptors = {"map_descriptors": desc}
    amosnet.set_map(test_descriptors)
    assert (amosnet.map_desc["map_descriptors"] == desc).all()

def test_set_descriptors(amosnet):
    desc = np.random.rand(10, 65)
    test_descriptors = {"query_descriptors": desc}
    amosnet.set_query(test_descriptors)
    amosnet.save_descriptors()

def test_save_descriptors(amosnet):
    desc = np.random.rand(10, 65)
    test_descriptors = {"query_descriptors": desc}
    amosnet.set_query(test_descriptors)
    desc = np.random.rand(20, 65)
    test_descriptors = {"map_descriptors": desc}
    amosnet.set_map(test_descriptors)
    amosnet.save_descriptors("tmp")

def test_load_descriptors(amosnet):
    query_desc = np.random.rand(10, 65)
    test_descriptors = {"query_descriptors": query_desc}
    amosnet.set_query(test_descriptors)
    map_desc = np.random.rand(20, 65)
    test_descriptors = {"map_descriptors":map_desc}
    amosnet.set_map(test_descriptors)
    amosnet.save_descriptors("tmp")
    amosnet.load_descriptors("tmp")
    assert (query_desc == amosnet.query_desc["query_descriptors"]).all()
    assert (map_desc == amosnet.query_desc["map_descriptors"]).all()

def test_compute_query_desc(dataset, amosnet):
    dataloader = dataset.query_images_loader(batch_size=10, preprocess=amosnet.preprocess)
    dataset = dataloader.dataset 
    dataset = Subset(dataset, list(range(10)))
    amosnet.compute_query_desc(dataloader=dataloader, pbar=False)

def test_compute_map_desc(dataset, amosnet):
    dataloader = dataset.map_images_loader(batch_size=10, preprocess=amosnet.preprocess)
    dataset = dataloader.dataset 
    dataset = Subset(dataset, list(range(10)))
    amosnet.compute_map_desc(dataloader=dataloader, pbar=False)



