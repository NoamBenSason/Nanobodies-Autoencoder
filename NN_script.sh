#!/bin/csh
#SBATCH --mem=5000m
#SBATCH -c2
#SBATCH --gres=gpu:1
#SBATCH --time=8:0:0


source /cs/usr/noam_bs97/3D-Hackton-SeqDesign/hacktonenv/bin/activate
module load tensorflow/2.5.0
module load cuda/11.3


cd /cs/usr/noam_bs97/3D-Hackton-SeqDesign
python3 train_network_wandb.py
