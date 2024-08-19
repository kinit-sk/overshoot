# overshoot

Stochastic gradient descent based optimization method for faster convergence.

## Requirements

 - Python packages: `pip install -r requirements.txt`
 - (Optional) Enviroment with GPU and cuda drivers

## Execution

To run experiments execute for example:
```
python train.py --job_name test --baseline --task_type roberta
```
When having cpu only it's recomended to use `--task_type cnn`.
For more info run `python train.py --help`.

## Execution on devana
To schedule jobs using slurm see: `devana-batch-jobs.sh`.
By default `devana-batch-jobs.sh` will run several jobs using various `overshoot-factors`
