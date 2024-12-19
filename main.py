import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from misc import (get_model_size, init_dataset, init_model, supported_datasets,
                  supported_models)
from optimizers import optimizers_map
from train import OvershootTrainer
from trainer_configs import get_trainer_config


def main():

    # 1) Create log writer
    base_dir = os.path.join("lightning_logs", args.experiment_name, args.job_name)
    os.makedirs(base_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=os.path.join(base_dir, f"version_{len(os.listdir(base_dir)) + 1}"))

    # 2) Create config
    trainer_config = get_trainer_config(args.model, args.dataset, args.opt_name, args.config_override)
    if trainer_config.precision == "high":
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_float32_matmul_precision("high")
    print("-------------------------------")
    print(f"Config: {trainer_config}")

    # 3) Create datatset
    dataset = init_dataset(args.dataset, args.model, args.seed)

    # 4) Create model
    model = init_model(args.model, dataset[0], trainer_config)
    if trainer_config.n_gpu > 0:
        model.cuda()
        # Doesn't work inside devana slurn job
        # model = torch.compile(model)
    print("-------------------------------")
    print(f"Model: {model}")
    print(f"Model size: {get_model_size(model)}")

    # 4) Launch trainer
    trainer = OvershootTrainer(model, dataset, log_writer, args, trainer_config)
    trainer.main()

    log_writer.close()


if __name__ == "__main__":
    # We should always observe the same results from:
    #   1) python main.py --seed 1 --config_override precision=high
    #   2) python main.py --seed 1 --two_models --overshoot_factor 0 --config_override precision=high
    # Sadly deterministic have to use 32-bit precision because of bug in pl.

    # We should observe the same results for:
    #  1)  python main.py --model mlp --dataset mnist --seed 1 --opt_name sgd_nesterov --config_override precision=high max_steps=160
    #  2)  python main.py --model mlp --dataset mnist --seed 1 --opt_name sgd_momentum --two_models --overshoot_factor 0.9 --config_override precision=high max_steps=160
    #  3)  python main.py --model mlp --dataset mnist --seed 1 --opt_name sgd_overshoot --overshoot_factor 0.9 --config_override precision=high max_steps=160
    # ADD 1: In case of nesterov only overshoot model is expected to be equal

    parser = argparse.ArgumentParser(
        """Train models using various custom optimizers.
                For baseline run:
                    `python main.py --model mlp --dataset mnist --opt_name sgd_momentun`
                Overshoot with two models implementation: 
                    `python main.py --model mlp --dataset mnist --opt_name sgd_momentum --two_models --overshoot_factor 3`
                Overshoot with efficient implementation: 
                    `python main.py --model mlp --dataset mnist --opt_name sgd_overshoot --overshoot_factor 3`
                To have deterministic results include: `--seed 42 --config_override precision=high`"""
    )
    parser.add_argument("--experiment_name", type=str, default="test", help="Folder name to store experiment results")
    parser.add_argument("--job_name", type=str, default="test", help="Sub-folder name to store experiment results")
    parser.add_argument("--overshoot_factor", type=float, help="Look-ahead factor when computng gradients")
    parser.add_argument(
        "--two_models",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use process with base and overshoot models",
    )
    parser.add_argument("--seed", type=int, required=False, help="If specified, use this seed for reproducibility.")
    parser.add_argument(
        "--opt_name", type=str, required=True, help=f"Supported optimizers are: {', '.join(optimizers_map.keys())}"
    )
    parser.add_argument("--compute_model_distance", action=argparse.BooleanOptionalAction, required=False)
    parser.add_argument(
        "--compute_base_model_loss", action=argparse.BooleanOptionalAction, required=False, default=True
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=f"Supported models are: {', '.join(supported_models)}",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=f"Supported datasets are: {', '.join(supported_datasets)}",
    )
    parser.add_argument(
        "--compute_cosine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute cosine similarity between successive vectors.",
    )
    parser.add_argument(
        "--config_override",
        type=str,
        nargs="+",
        default=None,
        help="Sequence of key-value pairs to override config. E.g. --config_override lr=0.01",
    )
    args = parser.parse_args()
    if args.seed:
        torch.manual_seed(args.seed)
    main()
