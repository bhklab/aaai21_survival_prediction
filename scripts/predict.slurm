#!/bin/bash
#SBATCH --job-name=aaai_predict
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -p gpu_radiomics
#SBATCH -A radiomics_gpu
#SBATCH --gres=gpu:1
#SBATCH -D /cluster/home/sejinkim/projects/aaai21_survival_prediction
#SBATCH --mail-user=sejin.kim@uhnresearch.ca
#SBATCH --mail-type=END
#SBATCH --output="/cluster/projects/radiomics/Temp/sejin/slurm_logs/%A-%x.out"

USERNAME="sejinkim"
source /cluster/home/$USERNAME/.bashrc
# OPTIONAL: activate the conda environment
conda activate aaai-survival

RAW_DATA_PATH=/cluster/projects/radiomics/RADCURE-challenge/data

python -m challenge_code.p2predict\
          $RAW_DATA_PATH\
          $RAW_DATA_PATH/test/clinical.csv\
          /cluster/home/sejinkim/projects/aaai21_survival_prediction/data/logs/aaai/dual_mtlr_1533071/checkpoints/epoch=249.ckpt\
          --num_workers 16\