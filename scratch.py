from PlaceRec.Methods import NetVLAD 
from PlaceRec.Datasets import StLucia_small
import numpy as np

ds = StLucia_small()
method = NetVLAD()

ground_truth = ds.ground_truth("train")
print(ground_truth)
