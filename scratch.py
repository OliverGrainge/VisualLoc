from PlaceRec.Datasets import GardensPointWalking, SFU, StLucia_small
import numpy as np

method = GardensPointWalking()

method = SFU()

from PlaceRec.utils import get_dataset


ds = "nordlands_spring"
ds = get_dataset(ds)

gt = ds.ground_truth("train", gt_type="hard")

print((gt.sum(0) == 0).any())
print((gt.sum(0) > 1).any())
