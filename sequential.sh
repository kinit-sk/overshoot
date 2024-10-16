#!/bin/bash
set -xe

# 1) Args
EXPERIMENT_NAME=${1:?"Missing experiment name"}
MODEL=${2:?"Missing model name. Options: 'gpt', 'cnn', 'gpt_hf', 'roberta_hf', 'bloom_hf', 'mdeberta_hf', 't5_hf'."}
DATASET=${3:?"Missing dataset name. a) vision: 'mnist', 'cifar100'. b) next-token-prediction: 'shakespear', 'gutenberg'. c) text-classification: 'qqp', 'mnli', 'mmlu'"}
N_RUNS=${4:-1}


# 2) Set seeds
SEEDS=()
for ((i=1; i<=${N_RUNS}; i++)); do
    SEEDS+=(${RANDOM})
done
# SEEDS="${SEEDS[@]}"
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
cp cnn.py "lightning_logs/${EXPERIMENT_NAME}/"
cp mlp.py "lightning_logs/${EXPERIMENT_NAME}/"
cp gpt.py "lightning_logs/${EXPERIMENT_NAME}/"
cp sequential.sh "lightning_logs/${EXPERIMENT_NAME}/"


# 3) Prepare output directory
CONFIG="--config_override epochs=1500 max_steps=5000 log_every_n_steps=10"
LRS=(0.0005 0.001 0.002 0.004 0.008 0.016)

for SEED in "${SEEDS[@]}"; do
    # for LR in "${LRS[@]}"; do
    #     CONFIG_FINAL="${CONFIG} lr_base=${LR}"
    #     python train.py ${PYTHON_ARGS_BASE} --job_name "lr_${LR}" --opt_name sgd_momentum --baseline --seed ${SEED} ${CONFIG_FINAL}
    # done
        
     
    python train.py ${PYTHON_ARGS_BASE} --job_name "baseline" --opt_name sgd_momentum --baseline --seed ${SEED} ${CONFIG}
    python train.py ${PYTHON_ARGS_BASE} --job_name "nesterov" --opt_name sgd_nesterov --baseline --seed ${SEED} ${CONFIG}
    
    
    OVERSHOOT="0.9"
    python train.py ${PYTHON_ARGS_BASE} --job_name "overshoot_${OVERSHOOT}_fast" --opt_name sgd_overshoot --overshoot_factor ${OVERSHOOT} --baseline --seed ${SEED} ${CONFIG}
    
    OVERSHOOT="3"
    python train.py ${PYTHON_ARGS_BASE} --job_name "overshoot_${OVERSHOOT}_fast" --opt_name sgd_overshoot --overshoot_factor ${OVERSHOOT} --baseline --seed ${SEED} ${CONFIG}
    
    OVERSHOOT="5"
    python train.py ${PYTHON_ARGS_BASE} --job_name "overshoot_${OVERSHOOT}_fast" --opt_name sgd_overshoot --overshoot_factor ${OVERSHOOT} --baseline --seed ${SEED} ${CONFIG}
    
    OVERSHOOT="9"
    python train.py ${PYTHON_ARGS_BASE} --job_name "overshoot_${OVERSHOOT}_fast" --opt_name sgd_overshoot --overshoot_factor ${OVERSHOOT} --baseline --seed ${SEED} ${CONFIG}
    
    OVERSHOOT="19"
    python train.py ${PYTHON_ARGS_BASE} --job_name "overshoot_${OVERSHOOT}_fast" --opt_name sgd_overshoot --overshoot_factor ${OVERSHOOT} --baseline --seed ${SEED} ${CONFIG}
    
    
done  