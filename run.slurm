#!/bin/bash
#SBATCH --job-name=optuna-run
#SBATCH --output=logs/optuna-%j.out
#SBATCH --error=logs/optuna-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pi-cdonnat
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=06:00:00

# Initialize conda and activate environment
source /home/tcastellani/miniconda3/etc/profile.d/conda.sh
conda activate py11

# Run your command
python /home/tcastellani/sleepstages/augment_search.py
