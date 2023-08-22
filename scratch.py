
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

from PlaceRec.Methods import NetVLAD, CONVAP, MixVPR
from PlaceRec.Datasets import GardensPointWalking

method = NetVLAD()
ds = GardensPointWalking()
loader = ds.query_images_loader("test", preprocess=method.preprocess)
q_desc = method.compute_query_desc(dataloader=loader)


method = MixVPR()
ds = GardensPointWalking()
loader = ds.query_images_loader("test", preprocess=method.preprocess)
q_desc = method.compute_query_desc(dataloader=loader)



method = CONVAP()
ds = GardensPointWalking()
loader = ds.query_images_loader("test", preprocess=method.preprocess)
q_desc = method.compute_query_desc(dataloader=loader)
