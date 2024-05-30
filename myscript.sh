#!/bin/bash

python train.py --method ResNet50_GeM --enable_progress_bar --aggregation_pruning_rate 1.0
python train.py --method ResNet50_GeM --enable_progress_bar --aggregation_pruning_rate 1.5
python train.py --method ResNet50_GeM --enable_progress_bar --aggregation_pruning_rate 2.0
python train.py --method ResNet50_GeM --enable_progress_bar --aggregation_pruning_rate 0.5
python train.py --method ResNet50_GeM --enable_progress_bar --aggregation_pruning_rate 0.0