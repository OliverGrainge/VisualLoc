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
#python train.py --backbone vgg16 --aggregation netvlad
#python train.py --backbone vgg16 --aggregation gem
#python train.py --backbone vgg16 --aggregation mac
#python train.py --backbone vgg16 --aggregation spoc
#python train.py --backbone resnet101 --aggregation mac

#python train.py --backbone efficientnet --aggregation netvlad 

#python train.py --backbone dinov2 --aggregation netvlad
#python train.py --backbone dinov2 --aggregation mixvpr
#python train.py --backbone dinov2 --aggregation gem
#python train.py --backbone dinov2 --aggregation spoc 
#python train.py --backbone dinov2 --aggregation mac



#======================================= Fine Tuning ========================================
DESCRIPTOR_SIZE=512
#python finetune.py --backbone resnet18 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE
#python finetune.py --backbone resnet50 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE
#python finetune.py --backbone vgg16 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE
#python finetune.py --backbone efficientnet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE 
#python finetune.py --backbone mobilenet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE 


#python finetune.py --backbone resnet18 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE 
#python finetune.py --backbone resnet50 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE
#python finetune.py --backbone vgg16 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE 
#python finetune.py --backbone efficientnet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE
#python finetune.py --backbone mobilenet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE


#python finetune.py --backbone resnet18 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE 
#python finetune.py --backbone resnet50 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE 
#python finetune.py --backbone vgg16 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE 
#python finetune.py --backbone efficientnet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE
#python finetune.py --backbone mobilenet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE



DESCRIPTOR_SIZE=2048
#python finetune.py --backbone resnet18 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE
#python finetune.py --backbone resnet50 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE
#python finetune.py --backbone vgg16 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE
#python finetune.py --backbone efficientnet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE 
#python finetune.py --backbone mobilenet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE 


#python finetune.py --backbone resnet18 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE 
#python finetune.py --backbone resnet50 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE
#python finetune.py --backbone vgg16 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE 
#python finetune.py --backbone efficientnet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE
#python finetune.py --backbone mobilenet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE


#python finetune.py --backbone resnet18 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE 
#python finetune.py --backbone resnet50 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE 
#python finetune.py --backbone vgg16 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE 
#python finetune.py --backbone efficientnet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE
#python finetune.py --backbone mobilenet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE



DESCRIPTOR_SIZE=4096
#python finetune.py --backbone resnet18 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE
#python finetune.py --backbone resnet50 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE
#python finetune.py --backbone vgg16 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE
#python finetune.py --backbone efficientnet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE 
#python finetune.py --backbone mobilenet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE 


python finetune.py --backbone resnet18 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE 
python finetune.py --backbone resnet50 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone vgg16 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE 
python finetune.py --backbone efficientnet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone mobilenet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE


python finetune.py --backbone resnet18 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE 
python finetune.py --backbone resnet50 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE 
python finetune.py --backbone vgg16 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE 
python finetune.py --backbone efficientnet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone mobilenet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE




python finetune.py --backbone dinov2 --aggregation mixvpr --descriptor_size 512
python finetune.py --backbone dinov2 --aggregation mixvpr --descriptor_size 2048
python finetune.py --backbone dinov2 --aggregation mixvpr --descriptor_size 4096

python finetune.py --backbone dinov2 --aggregation netvlad --descriptor_size 512
python finetune.py --backbone dinov2 --aggregation netvlad --descriptor_size 2048
python finetune.py --backbone dinov2 --aggregation netvlad --descriptor_size 4096

python finetune.py --backbone dinov2 --aggregation gem --descriptor_size 512
python finetune.py --backbone dinov2 --aggregation gem --descriptor_size 2048
python finetune.py --backbone dinov2 --aggregation gem --descriptor_size 4096

python finetune.py --backbone vgg16 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE 