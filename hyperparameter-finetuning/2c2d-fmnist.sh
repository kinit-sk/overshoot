#!/bin/bash
set -xe









# Setup
MODEL="2c2d"
DATASET="fmnist"
SEED="42"




LRS=(0.032 0.064)
# Intermediate variables
PYTHON_ARGS_BASE="--model ${MODEL} --dataset ${DATASET} --seed ${SEED}"
EXPERIMENT_BASE_NAME="hyperparameter-finetuning/${MODEL}-${DATASET}"

for LR in "${LRS[@]}"; do
    echo "Processing: ${LR}"
    opt_name="sgd_momentum"
    python main.py ${PYTHON_ARGS_BASE} --experiment_name "${EXPERIMENT_BASE_NAME}/${opt_name}" --job_name "${opt_name}-lr=${LR}" --opt_name ${opt_name} --config_override lr=${LR} use_lr_scheduler=True
done









# In paper used: 0.001 (however with no learning rate scheduler)
LRS=(0.0005 0.001 0.002 0.004 0.008 0.016 0.032 0.064)

# Intermediate variables
PYTHON_ARGS_BASE="--model ${MODEL} --dataset ${DATASET} --seed ${SEED}"
EXPERIMENT_BASE_NAME="hyperparameter-finetuning/${MODEL}-${DATASET}"

for LR in "${LRS[@]}"; do
    echo "Processing: ${LR}"
    opt_name="adamW"
    python main.py ${PYTHON_ARGS_BASE} --experiment_name "${EXPERIMENT_BASE_NAME}/${opt_name}" --job_name "${opt_name}-lr=${LR}" --opt_name ${opt_name} --config_override lr=${LR} use_lr_scheduler=True
done


