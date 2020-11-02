#!/bin/bash
#SBATCH --job-name=xd
#SBATCH --mem=16G
#SBATCH -t 2-00:00:00
#SBATCH -c 1
##SBATCH --account=radiomics_gpu
##SBATCH --partition=gpu_radiomics
##SBATCH --gres=gpu:1
#SBATCH -N 1

USERNAME="sejinkim"
source /cluster/home/$USERNAME/.bashrc
# OPTIONAL: activate the conda environment

echo start!
sleep 2h
scancel 1560124
scancel 1560125
scancel 1560129
scancel 1560130
scancel 1560134
scancel 1560135

echo 1st_half_clear!

sleep 6h
scancel 1560126
scancel 1560127
scancel 1560128
scancel 1560131
scancel 1560132
scancel 1560133
scancel 1560136
scancel 1560137
scancel 1560138

echo finished!