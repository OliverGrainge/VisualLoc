#!/bin/bash



DESCRIPTOR_SIZE=1024


PRECISION=fp32
BACKBONE=dinov2
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1

BACKBONE=vgg16
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1





PRECISION=fp16
BACKBONE=dinov2
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1

BACKBONE=vgg16
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1






PRECISION=int8
BACKBONE=dinov2
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1

BACKBONE=vgg16
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --precision $PRECISION --datasets pitts30k nordlands mapillarysls --metrics recall@1

















PRECISION=fp32
BACKBONE=mobilenet
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 512 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 2048 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 4096 --precision $PRECISION --datasets pitts30k

CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 512 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 2048 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 4096 --precision $PRECISION --datasets pitts30k

CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 512 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 2048 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 4096 --precision $PRECISION --datasets pitts30k

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 512 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 2048 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 4096 --precision $PRECISION --datasets pitts30k --metrics recall@1

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 512 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 2048 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 4096 --precision $PRECISION --datasets pitts30k --metrics recall@1

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 512 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 2048 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 4096 --precision $PRECISION --datasets pitts30k --metrics recall@1



BACKBONE=efficientnet
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 512 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 2048 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 4096 --precision $PRECISION --datasets pitts30k

CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 512 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 2048 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 4096 --precision $PRECISION --datasets pitts30k

CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 512 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 2048 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 4096 --precision $PRECISION --datasets pitts30k

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 512 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 2048 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 4096 --precision $PRECISION --datasets pitts30k --metrics recall@1

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 512 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 2048 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 4096 --precision $PRECISION --datasets pitts30k --metrics recall@1

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 512 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 2048 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 4096 --precision $PRECISION --datasets pitts30k --metrics recall@1


BACKBONE=resnet50
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 512 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 2048 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 4096 --precision $PRECISION --datasets pitts30k

CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 512 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 2048 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 4096 --precision $PRECISION --datasets pitts30k

CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 512 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 2048 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 4096 --precision $PRECISION --datasets pitts30k

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 512 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 2048 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 4096 --precision $PRECISION --datasets pitts30k --metrics recall@1

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 512 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 2048 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 4096 --precision $PRECISION --datasets pitts30k --metrics recall@1

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 512 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 2048 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 4096 --precision $PRECISION --datasets pitts30k --metrics recall@1








BACKBONE=vgg16
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 512 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 1024 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 2048 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 4096 --precision $PRECISION --datasets pitts30k

CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 512 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 1024 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 2048 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 4096 --precision $PRECISION --datasets pitts30k

CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 512 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 1024 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 2048 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 4096 --precision $PRECISION --datasets pitts30k

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 512 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 1024 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 2048 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 4096 --precision $PRECISION --datasets pitts30k --metrics recall@1

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 512 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 1024 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 2048 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 4096 --precision $PRECISION --datasets pitts30k --metrics recall@1

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 512 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 1024 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 2048 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 4096 --precision $PRECISION --datasets pitts30k --metrics recall@1




BACKBONE=dinov2
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 512 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 1024 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 2048 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 4096 --precision $PRECISION --datasets pitts30k

CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 512 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 1024 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 2048 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 4096 --precision $PRECISION --datasets pitts30k

CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 512 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 1024 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 2048 --precision $PRECISION --datasets pitts30k
CUDA_MODULE_LOADING=LAZY python run.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 4096 --precision $PRECISION --datasets pitts30k

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 512 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 1024 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 2048 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mac --descriptor_size 4096 --precision $PRECISION --datasets pitts30k --metrics recall@1

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 512 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 1024 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 2048 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation netvlad --descriptor_size 4096 --precision $PRECISION --datasets pitts30k --metrics recall@1

CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 512 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 1024 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 2048 --precision $PRECISION --datasets pitts30k --metrics recall@1
CUDA_MODULE_LOADING=LAZY python eval.py --method quantvpr --backbone $BACKBONE --aggregation mixvpr --descriptor_size 4096 --precision $PRECISION --datasets pitts30k --metrics recall@1