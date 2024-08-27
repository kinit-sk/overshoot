#!/bin/bash
# set -xe

# e.g., ./devana-batch-jobs.sh large_batch_size_test gpt_hf shakespear
JOB_NAME=${1:?"Missing experiment name"}
MODEL=${2:?"Missing model name. Options: 'gpt', 'cnn', 'gpt_hf', 'roberta_hf')."}
DATASET=${3:?"Missing dataset name. a) vision: 'mnist', 'cifar100'. b) next-token-prediction: 'shakespear', 'gutenberg'. c) text-classification: 'qqp', 'mnli', 'mmlu'"}

OVERSHOOT_FACTORS=(1.0 2.0 4.0 6.0 14.0 24.0)

DST="lightning_logs/${JOB_NAME}"
if [ -d "${DST}" ]; then
    echo "Removing previous experiments results!"
    rm -rf "${DST}"
fi
mkdir -p "${DST}"
mkdir -p "slurm_logs"

cp train.py "lightning_logs/${JOB_NAME}/"
cp custom_datasets.py "lightning_logs/${JOB_NAME}/"
cp cnn.py "lightning_logs/${JOB_NAME}/"
cp gpt.py "lightning_logs/${JOB_NAME}/"
cp trainer_configs.py "lightning_logs/${JOB_NAME}/"
cp devana-job.sh "lightning_logs/${JOB_NAME}/"
cp devana-batch-jobs.sh "lightning_logs/${JOB_NAME}/"

for FACTOR in "${OVERSHOOT_FACTORS[@]}"; do
    sbatch --output="slurm_logs/${JOB_NAME}___${MODEL_TYPE}___factor:_${FACTOR}.job" -J "${JOB_NAME}"  --export=ALL,JOB_NAME=${JOB_NAME},MODEL=${MODEL},DATASET=${DATASET},OVERSHOOT_FACTOR=${FACTOR} devana-job.sh 
done

# sbatch --output="slurm_logs/${JOB_NAME}___${MODEL_TYPE}___baseline.job" -J "${JOB_NAME}"  --export=ALL,JOB_NAME=${JOB_NAME},MODEL=${MODEL},DATASET=${DATASET} devana-job.sh 