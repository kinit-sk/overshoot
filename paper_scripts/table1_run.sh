#!/bin/bash

OVERSHOOT_FACTORS=(3 5 7)
SEEDS=(10 20 30 40 50 60 70 80 90 100)
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
EXPERIMENT_NAME="table1/mlp_housing_scheduler_longer"
MODEL="mlp"
DATASET="housing"
PYTHON_ARGS_BASE="--experiment_name ${EXPERIMENT_NAME} --model ${MODEL} --dataset ${DATASET} --config_override use_lr_scheduler=True"
copy_state "${EXPERIMENT_NAME}"


# SGD runs
PYTHON_ARGS="${PYTHON_ARGS_BASE} --job_name sgd_baseline --opt_name sgd_momentum"
sbatch --output="slurm_logs/sgd_baseline.job" -J "${EXPERIMENT_NAME}"  --export=ALL,PYTHON_ARGS="${PYTHON_ARGS}",SEEDS="${SEEDS}" devana-job-p.sh

PYTHON_ARGS="${PYTHON_ARGS_BASE} --job_name nesterov --opt_name sgd_nesterov"
sbatch --output="slurm_logs/nesterov.job" -J "${EXPERIMENT_NAME}"  --export=ALL,PYTHON_ARGS="${PYTHON_ARGS}",SEEDS="${SEEDS}" devana-job-p.sh

for FACTOR in "${OVERSHOOT_FACTORS[@]}"; do
    PYTHON_ARGS="${PYTHON_ARGS_BASE} --job_name sgd_overshoot_${FACTOR} --opt_name sgd_overshoot --overshoot_factor ${FACTOR}"
    sbatch --output="slurm_logs/sgd_overshoot_${FACTOR}.job" -J "${EXPERIMENT_NAME}"  --export=ALL,PYTHON_ARGS="${PYTHON_ARGS}",SEEDS="${SEEDS}" devana-job-p.sh
done





# Adam runs
PYTHON_ARGS="${PYTHON_ARGS_BASE} --job_name adam_baseline --opt_name adamW"
sbatch --output="slurm_logs/adam_baseline.job" -J "${EXPERIMENT_NAME}"  --export=ALL,PYTHON_ARGS="${PYTHON_ARGS}",SEEDS="${SEEDS}" devana-job-p.sh

PYTHON_ARGS="${PYTHON_ARGS_BASE} --job_name nadam --opt_name nadam"
sbatch --output="slurm_logs/nadam.job" -J "${EXPERIMENT_NAME}"  --export=ALL,PYTHON_ARGS="${PYTHON_ARGS}",SEEDS="${SEEDS}" devana-job-p.sh

for FACTOR in "${OVERSHOOT_FACTORS[@]}"; do
    PYTHON_ARGS="${PYTHON_ARGS_BASE} --job_name adam_overshoot_${FACTOR} --opt_name adamW_overshoot_delayed --overshoot_factor ${FACTOR}"
    sbatch --output="slurm_logs/adam_overshoot_${FACTOR}.job" -J "${EXPERIMENT_NAME}"  --export=ALL,PYTHON_ARGS="${PYTHON_ARGS}",SEEDS="${SEEDS}" devana-job-p.sh
done



