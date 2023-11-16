#!/bin/bash
#SBATCH --job-name=lr_extract
#SBATCH --exclude=matrix-1-12,matrix-0-24,matrix-1-4,matrix-2-13,matrix-1-8,matrix-0-38
#SBATCH --output=/home/kdoherty/spurge/data_release/logs/lr_extract/output_%j.log
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=russ_reserved
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32g
#SBATCH --array=0
 
source ~/anaconda3/etc/profile.d/conda.sh
conda activate data-release

cd ~/spurge/data_release

python extract_best_lr.py