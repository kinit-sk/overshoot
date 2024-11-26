#!/bin/bash
set -xe

# OVERSHOOT_FACTORS=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0)
OVERSHOOTS=(0 1 2 3 4 5 6 7 8 9 10)
OVERSHOOTS="${OVERSHOOTS[@]}"

# SEEDS=(10 20 30 40 50 60 70 80 90 100)
# SEEDS="${SEEDS[@]}"

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
EXPERIMENT_NAME="graph2/mlp_housing_test_p"
MODEL="mlp"
DATASET="housing"
PYTHON_ARGS_BASE="--experiment_name ${EXPERIMENT_NAME} --model ${MODEL} --dataset ${DATASET} --opt_name adamW --two_models --compute_model_distance --config_override max_steps=200"
copy_state "${EXPERIMENT_NAME}"



# # Run computational node in parallel
# sbatch --output="slurm_logs/model_distance.job" -J "${EXPERIMENT_NAME}"  --export=ALL,PYTHON_ARGS="${PYTHON_ARGS_BASE}",OVERSHOOTS="${OVERSHOOTS}" ./paper_scripts/graph2_job-p-overshoots.sh

# # Run in logging node in parallel
for overshoot in ${OVERSHOOTS}; do
    PYTHON_ARGS_FINAL="${PYTHON_ARGS_BASE} --job_name overshoot_${overshoot} --overshoot_factor ${overshoot} --seed 42 --config_override epochs=40"
    python main.py ${PYTHON_ARGS_FINAL} & # Launch processes in parallel
    
    # Wait to not conflict about gpu resources with previous process
    sleep 30
done

wait
echo "All Python processes have finished."
