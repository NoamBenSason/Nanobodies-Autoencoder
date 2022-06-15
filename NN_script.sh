#!/bin/csh
#SBATCH --mem=5000m
#SBATCH -c2
#SBATCH --gres=gpu:1
#SBATCH --time=8:0:0


source 
module load tensorflow/2.5.0
module load cuda/11.3


cd ...
python3 ...
