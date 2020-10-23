#!/bin/bash
#SBATCH --job-name=aaai_train
#SBATCH --mem=16G
#SBATCH -t 2-00:00:00
#SBATCH -c 4
#SBATCH -N 1

USERNAME="sejinkim"
source /cluster/home/$USERNAME/.bashrc
# OPTIONAL: activate the conda environment

echo "bye"