USERNAME="sejinkim"
source /cluster/home/$USERNAME/.bashrc
# OPTIONAL: activate the conda environment
conda activate radcure-challenge

ssh -N -f -R 6099:localhost:6099 h4huhnlogin2
tensorboard --port 6099 --logdir data/logs/

