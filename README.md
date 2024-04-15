# VisualLoc

VisualLoc is a library built to unify the research in the field of visual place recognition.
It is the case that many state of the art algorithms reported in the literature are deployed
on slightly different configured datasets, making it difficult to determine state of the art. 
This repositoy implements over 10 of the best Visual Place Techniques currently available along 
with 10 benchmarks so that their performances can be accurately compared. 

The repository also provides a framework to build, design, train, deploy and evaluate new 
visual place recognition methods, thereby serving the research communities in visual navigation 
and place recognition.



## Usage 
The two main concepts in this repository are "methods" and "datasets" each can be instantaited 
as objects and have a number of member functions useful for visual place recognition tasks.
A workflow for using a method for place recognition can be shown below. 


```python
# Import the method
from PlaceReck.Methods import DinoSalad
from PlaceRec.Datasets import Pitts30k

# Create an instance of the method
method = DinoSalad(pretrained=True)

# Create and instance of the dataset
dataset = Pitts30k()

# Compute the map descriptors that will be queried 
map_dataloader = dataset.map_images_loader(
        preprocess=method.preprocess,
        num_workers=16,
        pin_memory=True,
        batch_size=32,
    )

method.compute_map_desc(map_dataloader)


# Run Place Matching 
img = Image.open('sample_query.jpg')
idx, dist = method.place_recognise(img)
# idx is index of the matching image in the map dataset 
# dist is the match embedding distance from the query to the retrieved match
```

To evaluate a method on a dataset using standard metrics. You can 
perform as follows. 

```python
from PlaceReck.Methods import DinoSalad
from PlaceRec.Datasets import Pitts30k
from PlaceRec.Evaluate import Eval

dataset = Pitts30k()
method = DinoSalad(pretrained=True)
eval = Eval(method, dataset)

# first for efficiency compute all the matches 
eval.compute_all_matches()

# to get the recall@k metric where in this example k = 1
rec = eval.ratk(1)

# We can also get the efficiency metrics crucial for deployment 
lat = eval.matching_latency()
lat = eval.extraction_cpu_latency()
lat = eval.extraction_gpu_latency()

params = eval.count_params()
flops = eval.count_flops()
```

Whilst all these metrics are computed in native pytorch, which 
though excellent for experimentation and development may not 
be optimal for deployment. We can add some performance optimizations 
using commercial deep learning compilers, including DeepSparse 
and Nvidia's TensorRT. The usage has been made very simple and can 
be performed as follows 


```python
from PlaceReck.Methods import DinoSalad
from PlaceRec.Evaluate import Eval
from PlaceRec.Datasets import Pitts30k
from PlaceRec.Deploy import deploy_tensorrt_sparse, deploy_cpu_sparse

method = deploy_gpu(method, batch_size=1, sparse=True) # using Tensorrt backend
# or 
method = deploy_cpu(method, batch_size=1, sparse=True) # using the DeepSparse backend

dataset = Pitts30k()
eval = Eval(method, dataset)
lat = eval.matching_latency()
lat = eval.extraction_cpu_latency()
lat = eval.extraction_gpu_latency()
```


## Implementations 
Currently the following Methods are implemented in our framework. This is constantly updating and expanding as newer better methods are being released into the literature.

### Visual Place Recognition Techniques
1. HybridNet
2. AmosNet
3. CosPlace
4. EigenPlaces
5. MixVPR
6. AnyLoc
7. DinoSALAD
8. SelaVPR
9. SFRS
10. ConvAP
... We additionaly offer a number of different architectures
that can be trained with our trainers based on the excellent work 
of GSV-Cities, EigenPlaces and CosPlace

### Visual Place Recognition Datasets 
1. CrossSeasons
2. Essex3in1 
3. Gardens Point Walking 
4. Inria Holidays 
5. MapillarySLS (val)
6. Nordlands 
7. Pitts30k
8. Pitts250k
9. SFU 
10. SpedTest






