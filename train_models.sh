#!/bin/bash

python train.py --backbone resnet18 --aggregation mixvpr
python train.py --backbone resnet18 --aggregation netvlad
python train.py --backbone resnet18 --aggregation convap
python train.py --backbone resnet18 --aggregation gem
python train.py --backbone resnet18 --aggregation mac


python train.py --backbone resnet50 --aggregation mixvpr
python train.py --backbone resnet50 --aggregation netvlad
python train.py --backbone resnet50 --aggregation convap
python train.py --backbone resnet50 --aggregation gem
python train.py --backbone resnet50 --aggregation mac


python train.py --backbone resnet101 --aggregation mixvpr
python train.py --backbone resnet101 --aggregation netvlad
python train.py --backbone resnet101 --aggregation convap
python train.py --backbone resnet101 --aggregation gem
python train.py --backbone resnet101 --aggregation mac