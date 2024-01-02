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

@pytest.fixutre
def amosnet():
    return Methods.AmosNet()


