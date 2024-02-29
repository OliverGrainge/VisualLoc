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
    return Methods.AlexNet_AmosNet(pretrained=False)



@pytest.fixture
def cct_netvlad():
    return Methods.CCT_NetVLAD(pretrained=False)


@pytest.fixture
def resnet50convap():
    return Methods.ResNet50_ConvAP(pretrained=False)


@pytest.fixture
def cosplace():
    return Methods.ResNet50_CosPlace(pretrained=False)


@pytest.fixture
def hybridnet():
    return Methods.AlexNet_HybridNet(pretrained=False)


@pytest.fixture
def mixvpr():
    return Methods.ResNet50_MixVPR(pretrained=False)


@pytest.fixture
def netvlad():
    return Methods.ResNet18_NetVLAD(pretrained=False)


@pytest.fixture
def resnet18_gem():
    return Methods.ResNet18_GeM(pretrained=False)


@pytest.fixture
def resnet50_gem():
    return Methods.ResNet50_GeM(pretrained=False)


@pytest.fixture
def eigenplaces():
    return Methods.ResNet50_EigenPlaces(pretrained=False)
