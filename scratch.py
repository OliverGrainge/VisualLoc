
import boto3
import botocore

"""

bucket_name = 'visuallocbucket'
key = 'placerecdata/weights/msls_r18l3_netvlad_partial.pth'
download_path = './msls_r18l3_netvlad_partial.pth'

# Download the file


from PlaceRec import s3_bucket_download 

s3_bucket_download(key, download_path)

"""

from PlaceRec.Methods import NetVLAD, CONVAP, MixVPR, AmosNet, HybridNet, CALC
from PlaceRec.Datasets import GardensPointWalking

method = HybridNet()
ds = GardensPointWalking()
loader = ds.query_images_loader("test", preprocess=method.preprocess)
q_desc = method.compute_query_desc(dataloader=loader)


method = AmosNet()
ds = GardensPointWalking()
loader = ds.query_images_loader("test", preprocess=method.preprocess)
q_desc = method.compute_query_desc(dataloader=loader)



method = CALC()
ds = GardensPointWalking()
loader = ds.query_images_loader("test", preprocess=method.preprocess)
q_desc = method.compute_query_desc(dataloader=loader)
