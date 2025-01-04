#!/bin/bash
set -xe


EXPERIMENT_NAME="ci_run"
DST="lightning_logs/${EXPERIMENT_NAME}"
if [ -d "${DST}" ]; then
    echo "Removing previous experiments results!"
    rm -rf "${DST}"
fi
mkdir -p "${DST}"


MODEL="mlp"
DATASET="mnist"
SEED="1"
PYTHON_ARGS_BASE="--experiment_name ${EXPERIMENT_NAME} --model ${MODEL} --dataset ${DATASET} --seed ${SEED} --config_override precision=high max_steps=160"

#=== SGD
python main.py ${PYTHON_ARGS_BASE} --opt_name sgd_nesterov > /tmp/nesterov
python main.py ${PYTHON_ARGS_BASE} --opt_name sgd_momentum --two_models --overshoot_factor 0.9 > /tmp/two_models
python main.py ${PYTHON_ARGS_BASE} --opt_name sgd_overshoot --overshoot_factor 0.9 > /tmp/efficient

# Strip times
cat /tmp/nesterov | sed 's/ wall_time: [0-9]*.[0-9]* |//' | sed 's/ Time: [0-9]*.[0-9]*//g' > /tmp/nesterov_no_time
cat /tmp/two_models | sed 's/ wall_time: [0-9]*.[0-9]* |//' | sed 's/ Time: [0-9]*.[0-9]*//g' > /tmp/two_models_no_time
cat /tmp/efficient | sed 's/ wall_time: [0-9]*.[0-9]* |//' | sed 's/ Time: [0-9]*.[0-9]*//g' > /tmp/efficient_no_time

# 1) Test general and efficient implementation
diff /tmp/efficient_no_time /tmp/two_models_no_time > /dev/null
echo "SGD efficient vs two models OK"

# 2) Test efficient implementation vs nesterov
cat /tmp/nesterov_no_time | sed 's/ base_loss_1: [0-9]*.[0-9]* |//' | grep -v "Epoch" > /tmp/nesterov_no_base
cat /tmp/efficient_no_time | sed 's/ base_loss_1: [0-9]*.[0-9]* |//' | grep -v "Epoch" > /tmp/efficient_no_base
# diff /tmp/efficient_no_base /tmp/nesterov_no_base > /dev/null
diff /tmp/efficient_no_base /tmp/nesterov_no_base
echo "Efficient vs nesterov OK"



#=== Adam
python main.py ${PYTHON_ARGS_BASE} --opt_name adamW --two_models --overshoot_factor 10 > /tmp/two_models
python main.py ${PYTHON_ARGS_BASE} --opt_name adamW_overshoot_replication  --overshoot_factor 10 > /tmp/efficient

# Strip times
cat /tmp/two_models | sed 's/ wall_time: [0-9]*.[0-9]* |//' | sed 's/ Time: [0-9]*.[0-9]*//g' > /tmp/two_models_no_time
cat /tmp/efficient | sed 's/ wall_time: [0-9]*.[0-9]* |//' | sed 's/ Time: [0-9]*.[0-9]*//g' > /tmp/efficient_no_time

# 3) Test general and efficient implementation
diff /tmp/efficient_no_time /tmp/two_models_no_time > /dev/null
echo "AdamW efficient vs two models OK"

