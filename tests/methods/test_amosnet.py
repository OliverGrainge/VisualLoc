import pytest 


def test_name(amosnet):
    assert isinstance(amosnet.name, str)
    assert amosnet.name.islower()

def test_device(amosnet):
    assert amosnet.device in ["cpu", "cuda", "mps"]