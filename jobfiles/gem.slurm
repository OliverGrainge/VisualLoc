#!/bin/bash

#SBATCH --job-name=gem       # Job name
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=16            # Number of CPUs
#SBATCH --time=60:00:00
#SBATCH --gres=gpu:1 
#SBATCH --mem=64G                     # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=jobfiles/logs/gem.log           # Standard output log file (%j expands to jobId)
#SBATCH --error=jobfiles/logs/gem.err            # Standard error log file (%j expands to jobId)
#SBATCH --partition=ecsstaff              # Partition/queue name, replace 'defq' with your cluster's specific queue name if needed
#SBATCH --account=ecsstaff              # Partition/queue name, replace 'defq' with your cluster's specific queue name if needed

export WANDB_MODE=offline
python train.py --method ResNet34_GeM --milestones 1 2 3 --pruning_freq 2 --max_epochs 30 --aggregation_pruning_rate 0.25 --final_sparsity 0.75
python train.py --method ResNet34_GeM --milestones 1 2 3 --pruning_freq 2 --max_epochs 30 --aggregation_pruning_rate 0.0 --final_sparsity 0.75
python train.py --method ResNet34_GeM --milestones 1 2 3 --pruning_freq 2 --max_epochs 30 --aggregation_pruning_rate 0.5 --final_sparsity 0.75
python train.py --method ResNet34_GeM --milestones 1 2 3 --pruning_freq 2 --max_epochs 30 --aggregation_pruning_rate 0.75 --final_sparsity 0.75