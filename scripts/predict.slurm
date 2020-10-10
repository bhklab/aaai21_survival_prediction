#!/bin/bash
#SBATCH --job-name=simple_cnn
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -p gpu_radiomics
#SBATCH -A radiomics_gpu
#SBATCH --gres=gpu:1
#SBATCH -D /cluster/home/sejinkim/projects/uhn-radcure-challenge

USERNAME="sejinkim"
source /cluster/home/$USERNAME/.bashrc
# OPTIONAL: activate the conda environment
conda activate radcure-challenge

python -m radcurechallenge.phase2.p2predict\
        /cluster/home/sejinkim/projects/uhn-radcure-challenge/data/logs/ckpt_p2/dropout_1160397/checkpoints/simplecnn_epoch=32-val_loss=9.32e-01-roc_auc=0.83.ckpt\
        ./data/predictions/phase2.csv\
        /cluster/projects/radiomics/RADCURE-challenge/data/test\
        /cluster/projects/radiomics/RADCURE-challenge/data/test/clinical.csv\
        
