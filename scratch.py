from PlaceRec.Datasets import GardensPointWalking
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