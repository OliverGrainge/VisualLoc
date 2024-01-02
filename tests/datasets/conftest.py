
import pytest 
import os
from glob import glob
import sys

current_directory = os.getcwd()
new_directory = os.path.abspath(os.path.join(current_directory, '..', '..'))
sys.path.insert(0, new_directory)
os.chdir(new_directory)

from PlaceRec import Datasets


@pytest.fixture
def crossseasons():
    return Datasets.CrossSeason()

