#!/bin/bash
set -xe


# Setup
MODEL="mlp"
DATASET="housing"
SEED="1001"

## In paper used: 0.001 (however with no learning rate scheduler)
LRS=(0.001 0.002 0.004 0.008 0.016 0.032)
## In paper used: 64
BATCHES=(32 64 128)
## In paper used: 0.9
MOMENTUMS=(0.85 0.9 0.95)


# Intermediate variables
PYTHON_ARGS_BASE="--model ${MODEL} --dataset ${DATASET} --seed ${SEED}"
EXPERIMENT_BASE_NAME="hyperparameter-finetuning-advance/mlp-ca"


for BATCH in "${BATCHES[@]}"; do
    for LR in "${LRS[@]}"; do
        for MOMENTUM in "${MOMENTUMS[@]}"; do
            opt_name="sgd_momentum"
            job_name="${opt_name}-lr=${LR}-batch=${BATCH}-momentum=${MOMENTUM}"
            python main.py ${PYTHON_ARGS_BASE} --experiment_name "${EXPERIMENT_BASE_NAME}/${opt_name}" --job_name ${job_name} --opt_name ${opt_name} --config_override use_lr_scheduler=True lr=${LR} batch=${BATCH} sgd_momentum=${MOMENTUM} 
            
            opt_name="adamW"
            job_name="${opt_name}-lr=${LR}-batch=${BATCH}-momentum=${MOMENTUM}"
            python main.py ${PYTHON_ARGS_BASE} --experiment_name "${EXPERIMENT_BASE_NAME}/${opt_name}" --job_name ${job_name} --opt_name ${opt_name} --config_override use_lr_scheduler=True lr=${LR} batch=${BATCH} adam_beta1=${MOMENTUM} 
        done
    done
done

