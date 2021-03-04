#!/bin/bash
#SBATCH --job-name=toooon
#SBATCH --mem=32G
#SBATCH -t 1-00:00:00
#SBATCH -c 16
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gres=gpu:1
#SBATCH -D /cluster/home/sejinkim/projects/aaai21_survival_prediction
#SBATCH --mail-user=sejin.kim@uhnresearch.ca
#SBATCH --mail-type=ALL
#SBATCH --output="/cluster/projects/radiomics/Temp/sejin/slurm_logs/%A-%x.out"

USERNAME="sejinkim"
source /cluster/home/$USERNAME/.bashrc
# OPTIONAL: activate the conda environment
conda activate aaai-survival


RAW_DATA_PATH=/cluster/projects/radiomics/RADCURE-challenge/data

python -m challenge_code.p2tune\
          $RAW_DATA_PATH\
          $RAW_DATA_PATH/training/clinical.csv\
          --cache_dir ./data/data_cache\
          --num_workers 16\
          --batch_size 10\
          --lr 5e-5\
          --weight_decay 4e-5\
          --max_epochs 500\
          --gpus 1\
          --exp_name ckpt_cnn\
          --design pro\
