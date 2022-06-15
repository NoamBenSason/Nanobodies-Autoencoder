#!/bin/csh
#SBATCH --mem=5000m
#SBATCH -c2
#SBATCH --gres=gpu:1
#SBATCH --time=8:0:0


source /cs/usr/noam_bs97/PycharmProjects/pythonProject/venv/test/bin/activate.csh
module load cuda/11.4
module load cudnn/8.2.2

cd /cs/usr/noam_bs97/3D-Hackton-SeqDesign
python3 train_network_wandb.py
