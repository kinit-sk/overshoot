#!/bin/bash

# e.g., ./devana-batch-jobs.sh large_batch_size_test gpt_hf shakespear
EXPERIMENT_NAME=${1:?"Missing experiment name"}
MODEL=${2:?"Missing model name. Options: 'gpt', 'cnn', 'gpt_hf', 'roberta_hf', 'bloom_hf', 'mdeberta_hf', 't5_hf'."}
DATASET=${3:?"Missing dataset name. a) vision: 'mnist', 'cifar100'. b) next-token-prediction: 'shakespear', 'gutenberg'. c) text-classification: 'qqp', 'mnli', 'mmlu'"}
N_RUNS=${4:-1}

OVERSHOOT_FACTORS=(3 5 7 9)
PYTHON_ARGS_BASE="--experiment_name ${EXPERIMENT_NAME} --model ${MODEL} --dataset ${DATASET}"

SEEDS=()
for ((i=1; i<=${N_RUNS}; i++)); do
    SEEDS+=(${RANDOM})
done
SEEDS="${SEEDS[@]}"

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

# LRS=(0.00001 0.00002 0.00005 0.0001 0.0002)
# OPT_NAME="adamW"
# for LR in "${LRS[@]}"; do
#     PYTHON_ARGS="${PYTHON_ARGS_BASE} --job_name lr_${LR} --opt_name ${OPT_NAME} --baseline --config_override lr=${LR}"
#     sbatch --output="slurm_logs/config:_"${PYTHON_ARGS// /___}".job" -J "${EXPERIMENT_NAME}"  --export=ALL,PYTHON_ARGS="${PYTHON_ARGS}",SEEDS="${SEEDS}" devana-job.sh 
# done



OPT_NAME="adamW"

JOB_NAME="baseline"
PYTHON_ARGS="${PYTHON_ARGS_BASE} --job_name ${JOB_NAME} --opt_name ${OPT_NAME} --baseline --compute_model_distance --config_override epochs=1 log_every_n_steps=10"
sbatch --output="slurm_logs/config:_${EXPERIMENT_NAME}_${JOB_NAME}.job" -J "${EXPERIMENT_NAME}"  --export=ALL,PYTHON_ARGS="${PYTHON_ARGS}",SEEDS="${SEEDS}" devana-job.sh 

OPT_NAME="nadam"
JOB_NAME="nadam"
PYTHON_ARGS="${PYTHON_ARGS_BASE} --job_name ${JOB_NAME} --opt_name ${OPT_NAME} --baseline --compute_model_distance --config_override epochs=1 log_every_n_steps=10"
sbatch --output="slurm_logs/config:_${EXPERIMENT_NAME}_${JOB_NAME}.job" -J "${EXPERIMENT_NAME}"  --export=ALL,PYTHON_ARGS="${PYTHON_ARGS}",SEEDS="${SEEDS}" devana-job.sh 

# JOB_NAME="baseline_95"
# PYTHON_ARGS="${PYTHON_ARGS_BASE} --job_name ${JOB_NAME} --opt_name ${OPT_NAME} --baseline --config_override sgd_momentum=0.95 epochs=1"
# sbatch --output="slurm_logs/config:_${EXPERIMENT_NAME}_${JOB_NAME}.job" -J "${EXPERIMENT_NAME}"  --export=ALL,PYTHON_ARGS="${PYTHON_ARGS}",SEEDS="${SEEDS}" devana-job.sh 


OPT_NAME="adamW"
for FACTOR in "${OVERSHOOT_FACTORS[@]}"; do
    JOB_NAME="overshoot_${FACTOR}"
    PYTHON_ARGS="${PYTHON_ARGS_BASE} --job_name ${JOB_NAME} --opt_name ${OPT_NAME} --compute_model_distance --overshoot_factor ${FACTOR} --config_override epochs=1 log_every_n_steps=10"
    sbatch --output="slurm_logs/${EXPERIMENT_NAME}_${JOB_NAME}.job" -J "${EXPERIMENT_NAME}"  --export=ALL,PYTHON_ARGS="${PYTHON_ARGS}",SEEDS="${SEEDS}" devana-job.sh 
    
    # JOB_NAME="95_overshoot_${FACTOR}"
    # PYTHON_ARGS="${PYTHON_ARGS_BASE} --job_name ${JOB_NAME} --opt_name ${OPT_NAME} --baseline --overshoot_factor ${FACTOR} --config_override sgd_momentum=0.95 epochs=1"
    # sbatch --output="slurm_logs/${EXPERIMENT_NAME}_${JOB_NAME}.job" -J "${EXPERIMENT_NAME}"  --export=ALL,PYTHON_ARGS="${PYTHON_ARGS}",SEEDS="${SEEDS}" devana-job.sh 
done

