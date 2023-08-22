# VisualLoc
VisualLoc is a package with implementations of all of the most relevant state of the art visual place recognition methods. 
The methods are implemented within a modular framework that can be easily expanded to other techniques. Furthermore, to 
evaluate the performance of techniques a number of benchmark datasets have been provided. They too are in an extensible 
format with an automated testing framework. 

To Instantiate the datasets complete the following. The raw image data will automatically download. 

 ```python 
from PlaceRec.Datasets import GardensPointWalking 

dataset = GardensPointWalking()
 ```


To instantiate the methods, complete the following. Again any required model weights will be downloaded automatically.

 ```python 
from PlaceRec.Methods import NetVLAD

method = NetVLAD()
 ```


To apply the various techniques to a given data the following processing steps can be performed. 

 ```python
from PlaceRec.Datasets import GardensWalkingPoint
from PlaceRec.Methods import NetVLAD 

method = NetVLAD()
dataset = GardensPointWalking()

query_loader = dataset.query_images_loader("train", preprocess=method.preprocess)
map_loader = dataset.map_images_loader("train", preprocess=method.preprocess)

query_desc = method.compute_query_desc(dataloader=query_loader)
map_desc = method.compute_map_desc(dataloader=map_loader)

# to compute the similarity matrix
similarity_matrix = method.similarity_matrix(query_desc, map_desc)

# to perform place recognition 
idx, scores = method.place_recognition(dataloader=query_loader)

# idx: a numpy array where query image i is matched with refernce image idx[i] with a cosine distance of scores[i]

 ```