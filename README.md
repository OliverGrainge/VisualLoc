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

# get query and map loaders for batch inference on images
query_loader = dataset.query_images_loader("train", preprocess=method.preprocess)
map_loader = dataset.map_images_loader("train", preprocess=method.preprocess)

# compute the image descriptors
query_desc = method.compute_query_desc(dataloader=query_loader)
map_desc = method.compute_map_desc(dataloader=map_loader)

# to compute the similarity matrix
similarity_matrix = method.similarity_matrix(query_desc, map_desc)

# to perform place recognition 
idx, scores = method.place_recognition(dataloader=query_loader)

# idx: a numpy array where query image i is matched with refernce image idx[i] with a cosine distance of scores[i]

 ```

 The Implemented Visual Place Recognition Techniques Include; 

 - NetVLAD
 - HybridNet
 - AmosNet
 - CALC
 - CosPlace 
 - MixVPR
 - CONVAP
 - HOG
 - AlexNet

The following BenchMarked Datasets Include: 
- GardensPointWalking 
- ESSEX3IN1
- SFU
- StLucia_small
- GsvCities 
- Nordlands 


# Command Line Interface for evaluating method performance
to first compute the descriptors. In this command line instruction all variables are lowercase. "partition" variable can take the 
name of 'train', 'test', 'val' or 'all'. The "batchsize" variable is an integer indicating the batchsize. Here {dataset_names} can include 
a list of technique names all lower case. {methods_names} can again include multiple technique names in lower case

```console
python main.py --datasets {dataset_names} --methods {methods_names} --partition {partition} --batchsize {batchsize} --mode describe
```

To evaluate the performance of the various methods. The following command can be performed. It will create a folder /Plots containing 
a number of plots evaluating performance of vpr techniques.
```console
python main.py --datasets {dataset_name} --methods {methods_names} --partition {partition} --batchsize {batchsize} --mode evaluate
               --metrics prcurve count_flops count_params recall@1 recall@5 recall@10
```
