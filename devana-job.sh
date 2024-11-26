#!/bin/bash
#SBATCH --account=p904-24-3
#SBATCH --mail-user=<jakub.kopal@kinit.sk>
#SBATCH --time=20:00:00 # Estimate to increase job priority

## Nodes allocation
#SBATCH --partition=gpu
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
set -xe

eval "$(conda shell.bash hook)"
conda activate overshoot

for seed in ${SEEDS}; do
    PYTHON_ARGS_FINAL="${PYTHON_ARGS} --seed ${seed}"
    python main.py ${PYTHON_ARGS_FINAL}
done

