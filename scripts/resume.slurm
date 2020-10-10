#!/bin/bash
### Example script for submitting GPU jobs on H4H
#SBATCH --job-name=ckpt_p2
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH -c 16
#SBATCH -N 1
# add the name of the partition (e.g. -p gpu) and your group's GPU account
# (e.g. -A mygroup_gpu) below
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gres=gpu:1
# replace the path below with the directory with your code
#SBATCH -D /cluster/home/sejinkim/projects/uhn-radcure-challenge

USERNAME="sejinkim"
source /cluster/home/$USERNAME/.bashrc
# OPTIONAL: activate the conda environment
conda activate radcure-challenge

RAW_DATA_PATH=/cluster/projects/radiomics/RADCURE-challenge/data

python -m radcurechallenge.phase2.p2resume\
          $RAW_DATA_PATH\
          $RAW_DATA_PATH/training/clinical.csv\
          --cache_dir ./data/data_cache\
          --num_workers 16\
          --batch_size 10\
          --lr 5e-5\
          --weight_decay 4e-5\
          --max_epochs 500\
          --gpus 1\
          --exp_name ckpt_p2\
          --path /cluster/home/sejinkim/projects/uhn-radcure-challenge/data/logs/phase2/lite_1141080/checkpoints/phase2_epoch=52-val_loss=1.01e+00-roc_auc=0.83.ckpt\
