#!/bin/bash
#SBATCH --job-name=xd
#SBATCH --mem=16G
#SBATCH -t 2-00:00:00
#SBATCH -c 4
##SBATCH --account=radiomics_gpu
##SBATCH --partition=gpu_radiomics
##SBATCH --gres=gpu:1
#SBATCH -N 1

USERNAME="sejinkim"
source /cluster/home/$USERNAME/.bashrc
# OPTIONAL: activate the conda environment

echo start!
sleep 5h
echo here

scancel 1553711
scancel 1553713
scancel 1553714
scancel 1553715
scancel 1553716
scancel 1553718
scancel 1553719
scancel 1553722
scancel 1553724
scancel 1553725
scancel 1553726
scancel 1553727
