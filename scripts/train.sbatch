#!/bin/bash
#SBATCH --job-name=sft-vla0
#SBATCH --gpus=8
#SBATCH --cpus-per-task=112
#SBATCH --mem=1TB
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Load any necessary modules or activate environment here
# module load cuda/11.8
# . ~/miniforge3/bin/activate owa

export WANDB_PROJECT=vla0

cd ../
source .venv/bin/activate
accelerate launch --num_processes=8 scripts/train.py --config configs/vla0.yaml \
    --output_dir ./runs/vla0_sft-lr4e-5-gradclip --run-name vla0_sft-lr4e-5-gradclip \
    --max-grad-norm 1.0