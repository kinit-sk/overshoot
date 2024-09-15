#!/bin/bash
# set -xe

# e.g., ./devana-batch-jobs.sh large_batch_size_test gpt_hf shakespear
EXPERIMENT_NAME=${1:?"Missing experiment name"}
MODEL=${2:?"Missing model name. Options: 'gpt', 'cnn', 'gpt_hf', 'roberta_hf', 'bloom_hf', 'mdeberta_hf', 't5_hf'."}
DATASET=${3:?"Missing dataset name. a) vision: 'mnist', 'cifar100'. b) next-token-prediction: 'shakespear', 'gutenberg'. c) text-classification: 'qqp', 'mnli', 'mmlu'"}
N_RUNS=${4:-1}

OVERSHOOT_FACTORS=(1.0 1.9 4.0 6.0 14.0 24.0)
PYTHON_ARGS_BASE="--experiment_name ${EXPERIMENT_NAME} --model ${MODEL} --dataset ${DATASET}"

SEEDS=()
for ((i=1; i<=${N_RUNS}; i++)); do
    SEEDS+=(${RANDOM})
done

DST="lightning_logs/${EXPERIMENT_NAME}"
if [ -d "${DST}" ]; then
    echo "Removing previous experiments results!"
    rm -rf "${DST}"
fi
mkdir -p "${DST}"
mkdir -p "slurm_logs"

cp train.py "lightning_logs/${EXPERIMENT_NAME}/"
cp custom_datasets.py "lightning_logs/${EXPERIMENT_NAME}/"
cp cnn.py "lightning_logs/${EXPERIMENT_NAME}/"
cp gpt.py "lightning_logs/${EXPERIMENT_NAME}/"
cp trainer_configs.py "lightning_logs/${EXPERIMENT_NAME}/"
cp devana-job.sh "lightning_logs/${EXPERIMENT_NAME}/"
cp devana-batch-jobs.sh "lightning_logs/${EXPERIMENT_NAME}/"

OPT_NAME="adamW_overshoot"
for FACTOR in "${OVERSHOOT_FACTORS[@]}"; do
    PYTHON_ARGS="${PYTHON_ARGS_BASE} --job_name ${OPT_NAME}_overshoot_${FACTOR} --opt_name ${OPT_NAME} --baseline --overshoot_factor ${FACTOR}"
    sbatch --output="slurm_logs/"${PYTHON_ARGS// /___}".job" -J "${EXPERIMENT_NAME}"  --export=ALL,PYTHON_ARGS="${PYTHON_ARGS}",SEEDS=${SEEDS} devana-job.sh 
done


# OPT_NAME="adam"
# PYTHON_ARGS="${PYTHON_ARGS_BASE} --job_name ${OPT_NAME} --opt_name ${OPT_NAME} --overshoot_factor 1.9"
# sbatch --output="slurm_logs/"${PYTHON_ARGS// /___}".job" -J "${EXPERIMENT_NAME}"  --export=ALL,PYTHON_ARGS="${PYTHON_ARGS}",SEEDS=${SEEDS} devana-job.sh 

# OPT_NAME="nadam"
# PYTHON_ARGS="${PYTHON_ARGS_BASE} --job_name ${OPT_NAME} --opt_name ${OPT_NAME} --baseline"
# sbatch --output="slurm_logs/"${PYTHON_ARGS// /___}".job" -J "${EXPERIMENT_NAME}"  --export=ALL,PYTHON_ARGS="${PYTHON_ARGS}",SEEDS=${SEEDS} devana-job.sh 

# OPT_NAME="adamW_overshoot"
# PYTHON_ARGS="${PYTHON_ARGS_BASE} --job_name ${OPT_NAME} --opt_name ${OPT_NAME} --baseline --overshoot_factor 1.9"
# sbatch --output="slurm_logs/"${PYTHON_ARGS// /___}".job" -J "${EXPERIMENT_NAME}"  --export=ALL,PYTHON_ARGS="${PYTHON_ARGS}",SEEDS=${SEEDS} devana-job.sh 
