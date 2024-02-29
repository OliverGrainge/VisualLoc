#!/bin/bash


PRECISION=fp32
BACKBONE=mobilenet
#CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 1024 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 1024 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 1024 --precision $PRECISION --datasets pitts30k

#CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 1024 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 1024 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 1024 --precision $PRECISION --datasets pitts30k --metrics recall@1



BACKBONE=efficientnet
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 1024 --precision $PRECISION --datasets pitts30k 
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 1024 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 1024 --precision $PRECISION --datasets pitts30k

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 1024 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 1024 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 1024 --precision $PRECISION --datasets pitts30k --metrics recall@1



BACKBONE=resnet50
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 1024 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 1024 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 1024 --precision $PRECISION --datasets pitts30k


CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 1024 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 1024 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 1024 --precision $PRECISION --datasets pitts30k --metrics recall@1


