#!/bin/bash



#======================================= Fine Tuning ========================================
DESCRIPTOR_SIZE=512

python finetune.py --backbone resnet50 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone resnet50 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE

python finetune.py --backbone resnet18 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone resnet18 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE

python finetune.py --backbone mobilenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone mobilenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE

python finetune.py --backbone efficientnet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone efficientnet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE

DESCRIPTOR_SIZE=2048

python finetune.py --backbone resnet50 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone resnet50 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE

python finetune.py --backbone resnet18 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone resnet18 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE

python finetune.py --backbone mobilenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone mobilenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE

python finetune.py --backbone efficientnet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone efficientnet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE


DESCRIPTOR_SIZE=4096

python finetune.py --backbone resnet50 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone resnet50 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone resnet50 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone resnet50 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE

python finetune.py --backbone resnet18 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone resnet18 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone resnet18 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone resnet18 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE

python finetune.py --backbone mobilenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone mobilenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone mobilenet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone mobilenet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE

python finetune.py --backbone efficientnet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone efficientnet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone efficientnet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE
python finetune.py --backbone efficientnet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE