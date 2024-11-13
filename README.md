# overshoot

Stochastic gradient descent based optimization method for faster convergence.

## Requirements

 - Python packages: `pip install -r requirements.txt`
 - (Optional) Enviroment with GPU and cuda drivers

## Run

To run baseline:
```
python train.py --model mlp --dataset mnist --opt_name sgd_momentun
```
To run overshoot with two models implementation:
```
python train.py --model mlp --dataset mnist --opt_name sgd_momentum --two_models --overshoot_factor 3
```
To run overshoot with efficient implementation:
```
python train.py --model mlp --dataset mnist --opt_name sgd_overshoot --overshoot_factor 3
```
To observe the same results include: `--seed 42 --high_precision`.

For detailed description of the args training entry-point run:
```
python train.py --help
```

## Monitor experiments
To observe training statistics when neither `experiment_name` nor `job_name` is specified run:
```
tensorboard --logdir lightning_logs/test/test --port 6006
```
In the browser open `localhost:6006`.

## Execution on devana
To schedule jobs using slurm see: `devana-batch-jobs.sh`.

