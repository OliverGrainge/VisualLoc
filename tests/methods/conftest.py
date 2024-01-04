import os
import sys

import pytest

sys.path.insert(0, os.getcwd())
os.chdir(os.getcwd())


from PlaceRec import Methods
from PlaceRec.Datasets import CrossSeason


@pytest.fixture
def dataset():
    return CrossSeason()


@pytest.fixture
def amosnet():
    return Methods.AmosNet(pretrained=False)


@pytest.fixture
def calc():
    return Methods.CALC(pretrained=False)


@pytest.fixture
def cct_netvlad():
    return Methods.CCT384_NetVLAD(pretrained=False)


@pytest.fixture
def resnet50convap():
    return Methods.ResNet50ConvAP(pretrained=False)


@pytest.fixture
def resnet34convap():
    return Methods.ResNet34ConvAP(pretrained=False)


@pytest.fixture
def cosplace():
    return Methods.CosPlace(pretrained=False)


@pytest.fixture
def hybridnet():
    return Methods.HybridNet(pretrained=False)


@pytest.fixture
def mixvpr():
    return Methods.MixVPR(pretrained=False)


@pytest.fixture
def netvlad():
    return Methods.NetVLAD(pretrained=False)


@pytest.fixture
def resnet18_gem():
    return Methods.ResNet18GeM(pretrained=False)


@pytest.fixture
def resnet50_gem():
    return Methods.ResNet50GeM(pretrained=False)
