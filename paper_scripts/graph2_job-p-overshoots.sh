#!/bin/bash
#SBATCH --account=p904-24-3
#SBATCH --mail-user=<jakub.kopal@kinit.sk>
#SBATCH --time=09:00:00 # Estimate to increase job priority

## Nodes allocation
#SBATCH --partition=gpu
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
set -xe

eval "$(conda shell.bash hook)"
conda activate overshoot

for overshoot in ${OVERSHOOTS}; do
    PYTHON_ARGS_FINAL="${PYTHON_ARGS} --job_name overshoot_${overshoot} --overshoot_factor ${overshoot} --seed 42 --config_override max_steps=200"
    python main.py ${PYTHON_ARGS_FINAL} & # Launch processes in parallel
    
    # Wait to not conflict about gpu resources with previous process
    sleep 30
done

wait
echo "All Python processes have completed."