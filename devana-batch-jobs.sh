#!/bin/bash
# set -xe

JOB_NAME=${1:?"Missing experiment name"}
TASK_TYPE=${2:?"Missing task type (e.g., 'gpt', 'cnn')."}

OVERSHOOT_FACTORS=(1.0 2.0 4.0 6.0 10.0 16.0 24.0)

DST="lightning_logs/${JOB_NAME}"
if [ -d "${DST}" ]; then
    echo "Removing previous experiments results!"
    rm -rf "${DST}"
fi
mkdir -p "${DST}"
mkdir -p "slurm_logs"

cp train.py "lightning_logs/${JOB_NAME}/"
cp datasets.py "lightning_logs/${JOB_NAME}/"
cp cnn.py "lightning_logs/${JOB_NAME}/"
cp gpt.py "lightning_logs/${JOB_NAME}/"
cp devana-job.sh "lightning_logs/${JOB_NAME}/"
cp devana-batch-jobs.sh "lightning_logs/${JOB_NAME}/"

for FACTOR in "${OVERSHOOT_FACTORS[@]}"; do
    sbatch --output="slurm_logs/${JOB_NAME}___${TASK_TYPE}___factor:_${FACTOR}.job" -J "${JOB_NAME}"  --export=ALL,JOB_NAME=${JOB_NAME},TASK_TYPE=${TASK_TYPE},OVERSHOOT_FACTOR=${FACTOR} devana-job.sh 
done

sbatch --output="slurm_logs/${JOB_NAME}___${TASK_TYPE}___baseline.job" -J "${JOB_NAME}"  --export=ALL,JOB_NAME=${JOB_NAME},TASK_TYPE=${TASK_TYPE} devana-job.sh 