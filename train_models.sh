#!/bin/bash

#python train.py --backbone resnet18 --aggregation mixvpr
#python train.py --backbone resnet18 --aggregation netvlad
#python train.py --backbone resnet18 --aggregation convap
#python train.py --backbone resnet18 --aggregation gem
#python train.py --backbone resnet18 --aggregation mac
#python train.py --backbone resnet18 --aggregation spoc

#python train.py --backbone resnet50 --aggregation mixvpr
#python train.py --backbone resnet50 --aggregation netvlad
#python train.py --backbone resnet50 --aggregation gem
#python train.py --backbone resnet50 --aggregation mac
#python train.py --backbone resnet50 --aggregation spoc

#python train.py --backbone mobilenet --aggregation mixvpr
#python train.py --backbone mobilenet --aggregation netvlad
#python train.py --backbone mobilenet --aggregation gem
#python train.py --backbone mobilenet --aggregation mac
#python train.py --backbone mobilenet --aggregation spoc


#python train.py --backbone squeezenet --aggregation mixvpr
#python train.py --backbone squeezenet --aggregation netvlad
#python train.py --backbone squeezenet --aggregation gem
#python train.py --backbone squeezenet --aggregation mac
#python train.py --backbone squeezenet --aggregation spoc

#python train.py --backbone resnet101 --aggregation mixvpr
#python train.py --backbone resnet101 --aggregation netvlad
#python train.py --backbone resnet101 --aggregation gem


#python train.py --backbone resnet101 --aggregation mac
#python train.py --backbone resnet101 --aggregation spoc
#python train.py --backbone resnet18 --aggregation mac 
#python train.py --backbone resnet50 --aggregation mac
#python train.py --backbone mobilenet --aggregation mac 
#python train.py --backbone squeezenet --aggregation mac
#python train.py --back

#python train.py --backbone efficientnet --aggregation mixvpr
#python train.py --backbone efficientnet --aggregation netvlad
#python train.py --backbone efficientnet --aggregation gem
#python train.py --backbone efficientnet --aggregation mac
#python train.py --backbone efficientnet --aggregation spoc


#python train.py --backbone vgg16 --aggregation mixvpr
python train.py --backbone vgg16 --aggregation netvlad
python train.py --backbone vgg16 --aggregation gem
python train.py --backbone vgg16 --aggregation mac
python train.py --backbone vgg16 --aggregation spoc
python train.py --backbone resnet101 --aggregation mac

python train.py --backbone efficientnet --aggregation netvlad # this needs to be run independently


