import os
import sys
from glob import glob

import pytest

sys.path.insert(0, os.getcwd())
os.chdir(os.getcwd())


from PlaceRec import Datasets


@pytest.fixture
def crossseasons():
    return Datasets.CrossSeason()


@pytest.fixture
def essex3in1():
    return Datasets.ESSEX3IN1()


@pytest.fixture
def gardenspointwalking():
    return Datasets.GardensPointWalking()


@pytest.fixture
def inriaholidays():
    return Datasets.InriaHolidays()


@pytest.fixture
def nordlands():
    return Datasets.Nordlands()


@pytest.fixture
def pitts30k():
    return Datasets.Pitts30k()

@pytest.fixture
def pitts250k():
    return Datasets.Pitts250k()


@pytest.fixture
def sfu():
    return Datasets.SFU()


@pytest.fixture
def spedtest():
    return Datasets.SpedTest()
