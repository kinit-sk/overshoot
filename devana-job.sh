#!/bin/bash
#SBATCH --account=p365-23-1
#SBATCH --mail-user=<jakub.kopal@kinit.sk>
##SBATCH --time=00:10:00 # Estimate to increase job priority

## Nodes allocation
#SBATCH --partition=gpu
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
set -xe

eval "$(conda shell.bash hook)"
conda activate overshoot



if [ -z ${OVERSHOOT_FACTOR+x} ]; then 
    python train.py --job_name ${JOB_NAME} --task_type ${TASK_TYPE} --baseline --deterministic
else
    python train.py --job_name ${JOB_NAME} --task_type ${TASK_TYPE} --overshoot_factor ${OVERSHOOT_FACTOR} --deterministic
fi
