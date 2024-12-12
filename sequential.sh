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


# OVERSHOOT=(3 5 7)
# for SEED in "${SEEDS[@]}"; do
#     python main.py ${PYTHON_ARGS_BASE} --job_name sgd_momentum --opt_name sgd_momentum  --seed ${SEED} --config_override log_every_n_steps=4
#     python main.py ${PYTHON_ARGS_BASE} --job_name sgd_overshoot_3 --opt_name sgd_overshoot --overshoot_factor 3 --seed ${SEED} --config_override log_every_n_steps=4
#     python main.py ${PYTHON_ARGS_BASE} --job_name sgd_overshoot_5 --opt_name sgd_overshoot --overshoot_factor 5 --seed ${SEED} --config_override log_every_n_steps=4
#     python main.py ${PYTHON_ARGS_BASE} --job_name sgd_overshoot_7 --opt_name sgd_overshoot --overshoot_factor 7 --seed ${SEED} --config_override log_every_n_steps=4
    
    
#     python main.py ${PYTHON_ARGS_BASE} --job_name adamW --opt_name adamW  --seed ${SEED} --config_override log_every_n_steps=4
#     python main.py ${PYTHON_ARGS_BASE} --job_name adamW_overshoot_3 --opt_name adamW_overshoot_delayed --overshoot_factor 3 --seed ${SEED} --config_override log_every_n_steps=4
#     python main.py ${PYTHON_ARGS_BASE} --job_name adamW_overshoot_5 --opt_name adamW_overshoot_delayed --overshoot_factor 5 --seed ${SEED} --config_override log_every_n_steps=4
#     python main.py ${PYTHON_ARGS_BASE} --job_name adamW_overshoot_7 --opt_name adamW_overshoot_delayed --overshoot_factor 7 --seed ${SEED} --config_override log_every_n_steps=4
# done


SEED="10"

python main.py ${PYTHON_ARGS_BASE} --job_name no_sch_baseline --opt_name sgd_momentum  --seed ${SEED}  &
sleep 10

# python main.py ${PYTHON_ARGS_BASE} --job_name no_sch_overshoot_7_two  --opt_name sgd_momentum --two_models --overshoot_factor 7  --seed ${SEED} &
# sleep 10

python main.py ${PYTHON_ARGS_BASE} --job_name no_sch_overshoot_7  --opt_name sgd_overshoot_v2 --overshoot_factor 0.007  --seed ${SEED}  &
sleep 10



python main.py ${PYTHON_ARGS_BASE} --job_name yes_sch_baseline --opt_name sgd_momentum  --seed ${SEED} --config_override use_lr_scheduler=True &
sleep 10

# python main.py ${PYTHON_ARGS_BASE} --job_name yes_sch_overshoot_7_two  --opt_name sgd_momentum --two_models --overshoot_factor 7  --seed ${SEED} --config_override use_lr_scheduler=True &
# sleep 10

python main.py ${PYTHON_ARGS_BASE} --job_name yes_sch_overshoot_7  --opt_name sgd_overshoot_v2 --overshoot_factor 0.007  --seed ${SEED} --config_override use_lr_scheduler=True &
sleep 10




wait
echo "All Python processes have completed."


# python main.py ${PYTHON_ARGS_BASE} --job_name adamW_overshoot_3 --opt_name adamW_overshoot_delayed --overshoot_factor 3 --seed ${SEED} --config_override log_every_n_steps=4
# python main.py ${PYTHON_ARGS_BASE} --job_name adamW_overshoot_5 --opt_name adamW_overshoot_delayed --overshoot_factor 5 --seed ${SEED} --config_override log_every_n_steps=4
# python main.py ${PYTHON_ARGS_BASE} --job_name adamW_overshoot_7 --opt_name adamW_overshoot_delayed --overshoot_factor 7 --seed ${SEED} --config_override log_every_n_steps=4