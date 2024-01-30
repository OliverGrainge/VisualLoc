#!/bin/bash

python train.py --backbone resnet18 --aggregation netvlad
python train.py --backbone resnet18 --aggregation convap
python train.py --backbone resnet18 --aggregation cosplace
python train.py --backbone resnet18 --aggregation mixvpr
python train.py --backbone resnet18 --aggregation gem
python train.py --backbone resnet18 --aggregation spoc
python train.py --backbone resnet18 --aggregation mac