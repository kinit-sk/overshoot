#!/bin/bash

OVERSHOOT_FACTORS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
# OVERSHOOT_FACTORS=(0)
# OVERSHOOT_FACTORS=(15)
# OVERSHOOT_FACTORS=(11 12 13 14 15)
# SEEDS=(10 20 30 40 50 60 70 80 90 100)
# SEEDS=(10 20 30 40 50 60 70 80 90 100)
SEEDS=(10)
SEEDS="${SEEDS[@]}"

copy_state() {
    local name=$1
    DST="lightning_logs/${name}"
    if [ -d "${DST}" ]; then
        echo "Removing previous experiments results!"
        rm -rf "${DST}"
    fi
    mkdir -p "${DST}"
    mkdir -p "slurm_logs"

    cp train.py "lightning_logs/${name}/"
    cp custom_datasets.py "lightning_logs/${name}/"
    cp trainer_configs.py "lightning_logs/${name}/"
    cp devana-job.sh "lightning_logs/${name}/"
    cp devana-batch-jobs.sh "lightning_logs/${name}/"
}



# TODO: CHANGE!!!!!!!!!!!!
# 1) First test case
EXPERIMENT_NAME="graph2/mlp_housing_95"
MODEL="mlp"
DATASET="housing"
# PYTHON_ARGS_BASE="--experiment_name ${EXPERIMENT_NAME} --model ${MODEL} --dataset ${DATASET}"
# PYTHON_ARGS_BASE="--experiment_name ${EXPERIMENT_NAME} --model ${MODEL} --dataset ${DATASET} --opt_name adamW_overshoot_delayed --no-compute_base_model_loss --compute_model_distance"
PYTHON_ARGS_BASE="--experiment_name ${EXPERIMENT_NAME} --model ${MODEL} --dataset ${DATASET} --opt_name adamW_overshoot_delayed --compute_model_distance"
copy_state "${EXPERIMENT_NAME}"



for FACTOR in "${OVERSHOOT_FACTORS[@]}"; do
    PYTHON_ARGS="${PYTHON_ARGS_BASE} --job_name overshoot_${FACTOR} --overshoot_factor ${FACTOR} --config_override adam_beta1=0.95 epochs=120"
    sbatch --output="slurm_logs/adam_overshoot_${FACTOR}.job" -J "${EXPERIMENT_NAME}"  --export=ALL,PYTHON_ARGS="${PYTHON_ARGS}",SEEDS="${SEEDS}" devana-job.sh
done
