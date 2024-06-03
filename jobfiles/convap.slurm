#!/bin/bash

#SBATCH --job-name=convap       # Job name
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=16            # Number of CPUs
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1 
#SBATCH --mem=64G                     # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=jobfiles/logs/convap.log           # Standard output log file (%j expands to jobId)
#SBATCH --error=jobfiles/logs/convap.err            # Standard error log file (%j expands to jobId)
#SBATCH --partition=ecsstaff              # Partition/queue name, replace 'defq' with your cluster's specific queue name if needed
#SBATCH --account=ecsstaff              # Partition/queue name, replace 'defq' with your cluster's specific queue name if needed

export WANDB_MODE=offline
python train.py --method ConvAP --checkpoint