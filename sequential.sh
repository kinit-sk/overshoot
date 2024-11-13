#!/bin/bash
set -xe

# 1) Args
EXPERIMENT_NAME=${1:?"Missing experiment name"}
MODEL=${2:?"Missing model name. Options: 'gpt', 'cnn', 'gpt_hf', 'roberta_hf', 'bloom_hf', 'mdeberta_hf', 't5_hf'."}
DATASET=${3:?"Missing dataset name. a) vision: 'mnist', 'cifar100'. b) next-token-prediction: 'shakespear', 'gutenberg'. c) text-classification: 'qqp', 'mnli', 'mmlu'"}
N_RUNS=${4:-1}


SEEDS=()
for ((i=1; i<=${N_RUNS}; i++)); do
    SEEDS+=(${RANDOM})
done
PYTHON_ARGS_BASE="--experiment_name ${EXPERIMENT_NAME} --model ${MODEL} --dataset ${DATASET}"


# 3) Prepare output directory
DST="lightning_logs/${EXPERIMENT_NAME}"
if [ -d "${DST}" ]; then
    echo "Removing previous experiments results!"
    rm -rf "${DST}"
fi
mkdir -p "${DST}"
cp train.py "lightning_logs/${EXPERIMENT_NAME}/"
cp trainer_configs.py "lightning_logs/${EXPERIMENT_NAME}/"
cp custom_datasets.py "lightning_logs/${EXPERIMENT_NAME}/"


OVERSHOOT=(3 5 7)
for SEED in "${SEEDS[@]}"; do
    python train.py ${PYTHON_ARGS_BASE} --job_name baseline --opt_name sgd_momentum  --seed ${SEED} --baseline
    for FACTOR in "${OVERSHOOT[@]}"; do
        python train.py ${PYTHON_ARGS_BASE} --job_name overshoot_${FACTOR} --opt_name sgd_momentum --overshoot_factor ${FACTOR} --seed ${SEED}
    done
    
done  