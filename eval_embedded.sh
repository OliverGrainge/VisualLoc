#!/bin/bash
DESCRIPTOR_SIZE=1024
PRECISION=fp32

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet  --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION


CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION


DESCRIPTOR_SIZE=1024
PRECISION=fp16

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet  --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION


CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION



DESCRIPTOR_SIZE=1024
PRECISION=int8

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet  --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION


CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION












DESCRIPTOR_SIZE=512
PRECISION=fp32

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet  --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION


CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION


DESCRIPTOR_SIZE=1024
PRECISION=fp16

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet  --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION


CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION



DESCRIPTOR_SIZE=1024
PRECISION=int8

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet  --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION


CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION








DESCRIPTOR_SIZE=2048
PRECISION=fp32

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet  --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION


CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION


DESCRIPTOR_SIZE=1024
PRECISION=fp16

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet  --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION


CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION



DESCRIPTOR_SIZE=1024
PRECISION=int8

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet  --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION


CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION








DESCRIPTOR_SIZE=2048
PRECISION=fp32

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet  --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION


CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION


DESCRIPTOR_SIZE=1024
PRECISION=fp16

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet  --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION


CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION



DESCRIPTOR_SIZE=4096
PRECISION=int8

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet18 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet50 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone resnet101 --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone efficientnet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION

CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet  --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone mobilenet --aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION


CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation spoc --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation mac --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation gem --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet --aggregation netvlad --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION
CUDA_MODULE_LOADING=LAZY python eval.py --methods quantvpr --backbone squeezenet--aggregation mixvpr --descriptor_size $DESCRIPTOR_SIZE --datasets stlucia --metrics gpu_embedded_latency --precision $PRECISION