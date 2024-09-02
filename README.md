# overshoot

Stochastic gradient descent based optimization method for faster convergence.

## Requirements

 - Python packages: `pip install -r requirements.txt`
 - (Optional) Enviroment with GPU and cuda drivers

## Execution

To run experiments execute for example:
```
python train.py --job_name test --model gpt --dataset shakespear --overshoot_factor 2.0
```
When having cpu only it's recomended to use `--model_type cnn`.
For more info run `python train.py --help`.

## Execution on devana
To schedule jobs using slurm see: `devana-batch-jobs.sh`.
By default `devana-batch-jobs.sh` will run several jobs using various `overshoot-factors`


## Monitor experiments
To observe training statistics run 
```
tensorboard --logdir lightning_logs/{job_name} --port 6007
```
In browser open `localhost:6007`.
