#!/bin/bash
#SBATCH --job-name=aaai_train
#SBATCH --mem=16G
#SBATCH -t 2-00:00:00
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
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

python -m challenge_code.p2train\
          $RAW_DATA_PATH\
          $RAW_DATA_PATH/training/clinical.csv\
          --cache_dir ./data/data_cache\
          --num_workers 0\
          --batch_size 8\
          --lr 1e-4\
          --c1 10\
          --weight_decay 1e-4\
          --max_epochs 500\
          --gpus 1\
          --exp_name aaai\
          --design aaai_final\
          --dense_factor 1\
          --n_dense 3\
          --seed 1129\