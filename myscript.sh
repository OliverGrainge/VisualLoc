#!/bin/bash

python train.py --method ResNet18_NetVLAD --batch_size 80 --enable_progress_bar --max_epochs=30 --image_resolution 320 320 --training_method gsv_cities_dense --milestones 5 10 15 20 25 --lr 0.0002
python train.py --method ResNet18_GeM --batch_size 80 --enable_progress_bar --max_epochs=30 --image_resolution 320 320 --training_method gsv_cities_dense --milestones 5 10 15 20 25 --lr 0.0002 --checkpoint
python train.py --method ResNet50_GeM --batch_size 80 --enable_progress_bar --max_epochs=30 --image_resolution 320 320 --training_method gsv_cities_dense --milestones 5 10 15 20 25 --lr 0.0002