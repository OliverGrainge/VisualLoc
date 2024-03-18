#!/bin/bash

#SBATCH --job-name=resnet50_mixvpr       # Job name
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=16            # Number of CPUs
#SBATCH --gres=gpu:1 
#SBATCH --mem=64G                     # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=jobfiles/logs/resnet50_mixvpr.log           # Standard output log file (%j expands to jobId)
#SBATCH --error=jobfiles/logs/resnet50_mixvpr.err            # Standard error log file (%j expands to jobId)
#SBATCH --partition=a100              # Partition/queue name, replace 'defq' with your cluster's specific queue name if needed

python train.py --method resnet50_mixvpr --image_resolution 320 320