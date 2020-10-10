#!/bin/bash
### Example script for submitting GPU jobs on H4H
#SBATCH --job-name=phase2_train_cpu
#SBATCH --mem=32G
#SBATCH -t 1-00:00:00
#SBATCH -c 32
#SBATCH -N 1
#SBATCH --partition=himem
# replace the path below with the directory with your code
#SBATCH -D /cluster/home/sejinkim/projects/uhn-radcure-challenge

USERNAME="sejinkim"
source /cluster/home/$USERNAME/.bashrc
# OPTIONAL: activate the conda environment
conda activate radcure-challenge

RAW_DATA_PATH=/cluster/projects/radiomics/RADCURE-challenge/data

python -m radcurechallenge.models.train\
          $RAW_DATA_PATH\
          $RAW_DATA_PATH/training/clinical.csv\
          --cache_dir ./data/data_cache\
          --num_workers 16\
          --batch_size 10\
          --lr 1e-3\
          --weight_decay 1e-4\
          --max_epochs 500\
          --gpus 0\
          --exp_name new_cnn\
          --design pro\
