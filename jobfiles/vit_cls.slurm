#!/bin/bash

#SBATCH --job-name=vit_cls        # Job name
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=16            # Number of CPUs
#SBATCH --gres=gpu:1 
#SBATCH --mem=64G                     # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=jobfiles/logs/vit_cls.log           # Standard output log file (%j expands to jobId)
#SBATCH --error=jobfiles/logs/vit_cls.err            # Standard error log file (%j expands to jobId)
#SBATCH --partition=a100              # Partition/queue name, replace 'defq' with your cluster's specific queue name if needed

python train.py --method vit_cls --image_resolution 224 224