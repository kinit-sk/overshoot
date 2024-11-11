import argparse
import copy
import os
import re
import time
import random

import pytorch_lightning as pl
import numpy as np
import pandas as pd
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader

from optimizers.sgdo import SGDO
from optimizers.sgdo_adaptive import SGDO as SGDO_adaptive
from optimizers.adamw_overshoot_replication import AdamW as OvershootAdamW_replication
from optimizers.adamw_overshoot_full_approximation import AdamW as OvershootAdamW_full_approximation
from optimizers.adamw_overshoot_denom_approximation import AdamW as OvershootAdamW_denom_approximation
from optimizers.adamw_overshoot_delayed import AdamW as OvershootAdamW_delayed
from optimizers.adamw_overshoot_adaptive import AdamW as OvershootAdamW_adaptive


from misc import init_dataset, init_model, get_gpu_stats, compute_model_distance
from trainer_configs import get_trainer_config

# ------------------------------------------------------------------------------
torch.cuda.empty_cache()
torch.set_float32_matmul_precision("high")
# -----------------------------------------------------------------------------


class OvershootTrainer(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, dataset, config):
        super(OvershootTrainer, self).__init__()
        # Manual optimization: https://lightning.ai/docs/pytorch/stable/common/optimization.html
        self.automatic_optimization = not args.two_models
        self.eval_model = None
        self.base_model = model
        if args.two_models:
            self.overshoot_model = copy.deepcopy(model)
        self.train_dataset, self.val_dataset = dataset
        self.steps = int(round(config.B + config.epochs * len(self.train_dataset) // config.B // max(1, config.n_gpu)))
        print("-----------------------------------------------")
        print("Total training steps: ", self.steps)
        print(f"Epoch steps: {len(self.train_dataset) // (config.B * max(1, config.n_gpu))}")
        print("--")
        print(f"Train dataset size: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"Valid dataset size: {len(self.val_dataset)}")
        print("-----------------------------------------------")
        if config.max_steps:
            self.steps = min(self.steps, config.max_steps)
        self.config = config
        self.train_stats, self.val_stats = [], []
        self.current_step = 0
        self.train_losses, self.train_accuracy = [], []
        self.val_losses, self.val_accuracy = [], []
        self.overshoot_losses = []
        self.all_params_base = []
        self.all_params_overshoot = []
        self.distances = []
        self.last_time = time.time()
        # Cosine gradient statistics
        self.previous_params, self.previous_params_est = None, None
        self.last_update, self.last_update_est = None, None
        self.update_cosine, self.update_cosine_est = [0], [0]

    def _compute_model_distance(self):
        latest_base_model = torch.cat([p.data.view(-1).cpu() for p in self.base_model.parameters()])
        if self.automatic_optimization:
            self.past_models.append(latest_base_model)
        else:
            self.past_models.append(torch.cat([p.data.view(-1).cpu() for p in self.overshoot_model.parameters()]))
        if len(self.past_models) > 50:
            self.past_models.pop(0)
        momentum = self.config.sgd_momentum if "sgd" in args.opt_name else self.config.adam_beta1
        return compute_model_distance(latest_base_model, self.past_models, momentum)
            
    def _cosine_similarity(self):
        sample_size = 1000
        params = torch.cat([p.data.view(-1) for p in self.base_model.parameters()])
        if not hasattr(self, "random_indices"):
            self.random_indices = torch.randint(0, params.size(0), (sample_size,))
        params_est = params[self.random_indices]
        if self.previous_params is not None:
            update = params - self.previous_params
            update_est = params_est - self.previous_params_est
            if self.last_update is not None:
                similarity = F.cosine_similarity(self.last_update, update, dim=0)
                similarity_est = F.cosine_similarity(self.last_update_est, update_est, dim=0)
                for arr, new in [(self.update_cosine, similarity), (self.update_cosine_est, similarity_est)]:
                    if not torch.isnan(new):
                        arr.append(new.item())
                        if len(arr) > 100:
                            arr.pop(0)
            self.last_update = update
            self.last_update_est = update_est
            
        self.previous_params = params
        self.previous_params_est = params_est

    # This just prints stats to console. Shouldn't be this complicated
    def __print_stats(self, stats):
        if stats["batch_step"] % self.config.log_every_n_steps == 0:
            k_v_to_str = lambda k, v: f'{k}: {round(v, 4) if type(v) == float else v}'
            text = ' | '.join([k_v_to_str(k, v) for k, v in stats.items() if not re.search(r"(loss|accuracy|similarity|est)_[0-9][0-9]+$", k)])
            print(text + (get_gpu_stats(self.config.n_gpu) if self.config.log_gpu else ''), flush=True)

    def _baseline_training_step(self, batch, batch_idx):
        # Compute base loss when having only overshoot models (slow).
        if hasattr(self.optimizers(), "move_to_base"):
            with torch.no_grad():
                self.optimizers().move_to_base()
                # !!! For some reason when performing the inference on base_model, training breaks
                # This deep copy is only needed when using GPT with 16-bit precision
                base_output = copy.deepcopy(self.base_model).forward(**batch)
                self.optimizers().move_to_overshoot()
        output = self.base_model.forward(**batch)
        if self.config.decay_lr:
            self.base_scheduler.step()
        if hasattr(self.optimizers(), "move_to_base"):
            return base_output["loss"], output["loss"], base_output["logits"]
        else:
            return output["loss"], output["loss"], output["logits"]


    def _overshoot_training_step(self, batch, batch_idx):
        with torch.no_grad():
            output_base = self.base_model.forward(**batch)  # only to log base loss
        output_overshoot = self.overshoot_model.forward(**batch)
        self.manual_backward(output_overshoot["loss"] / self.config.accumulate_grad_batches)

        if (batch_idx + 1) % self.config.accumulate_grad_batches == 0:

            # 1) Gradients OVERSHOOT -> BASE
            for param1, param2 in zip(self.overshoot_model.parameters(), self.base_model.parameters()):
                if param1.grad is not None:
                    param2.grad = param1.grad.clone()

            # 2) Weights BASE -> OVERSHOOT
            for param1, param2 in zip(self.base_model.parameters(), self.overshoot_model.parameters()):
                param2.data = param1.data.clone()

            # 3) (Optional) Update learning rates
            if self.config.decay_lr:
                self.base_scheduler.step()
                self.overshoot_scheduler.step()

            # 4) Update models based on gradients
            for opt in self.optimizers():
                opt.step()
                opt.zero_grad()

        return output_base["loss"], output_overshoot["loss"], output_base["logits"]

    def training_step(self, batch, batch_idx):

        # We compute model distances before model update to have the same behaviour for baseline and overshoot
        if args.compute_model_distance:
            model_distance = self._compute_model_distance()
            
        if args.compute_cosine:
            self._cosine_similarity()

            
        train_fn = self._baseline_training_step if self.automatic_optimization else self._overshoot_training_step
        loss_base, loss_overshoot, output_base = train_fn(batch, batch_idx)

        for losses, new_loss in [(self.train_losses, loss_base.item()), (self.overshoot_losses, loss_overshoot.item())]:
            losses.append(new_loss)
            if len(losses) > 100:
                losses.pop(0)
            
        stats = {
            "step": self.current_step,
            "epoch": self.current_epoch,
            "batch_step": batch_idx,
            "base_lr": self.base_scheduler.get_last_lr()[-1],
            "overshoot_lr": self.base_scheduler.get_last_lr()[-1] if self.automatic_optimization else self.overshoot_scheduler.get_last_lr()[-1],
            "td": time.time() - self.last_time,
        }
        for avg in [1, 20, 50, 100]:
            stats[f"base_loss_{avg}"] = float(np.mean(self.train_losses[-avg:]))
            stats[f"overshoot_loss_{avg}"] = float(np.mean(self.overshoot_losses[-avg:])) # For baseline same as base loss

        
        if hasattr(self.optimizers(), "_overshoot_new"):
            stats["overshoot_factor"] = self.optimizers()._overshoot_new
                
        if self.train_dataset.is_classification():
            self.train_accuracy.append(100 * torch.mean(output_base.argmax(dim=-1) == batch["labels"], dtype=float).item())
            if len(self.train_accuracy) > 100:
                self.train_accuracy.pop(0)
            for avg in [1, 20, 50, 100]:
                stats[f"accuracy_{avg}"] = float(np.mean(self.train_accuracy[-avg:]))

        if args.compute_cosine:
            for avg in [1, 20, 50, 100]:
                stats[f"update_cosine_similarity_{avg}"] = float(np.mean(self.update_cosine[-avg:]))
                stats[f"update_cosine_similarity_est_{avg}"] = float(np.mean(self.update_cosine_est[-avg:]))
            
        if args.compute_model_distance:
            stats["model_distance"] = model_distance
            
        self.__print_stats(stats)
        self.log_dict(stats)
        self.train_stats.append(stats)
        
        self.last_time = time.time()
        self.current_step += 1
        self.trainer.should_stop = self.current_step >= self.steps
        return loss_overshoot

    def validation_step(self, batch, batch_idx):
        if self.eval_model is None:
            if hasattr(self.optimizers(), "move_to_base"):
                with torch.no_grad():
                    self.optimizers().move_to_base()
                    self.eval_model = copy.deepcopy(self.base_model)
                    self.optimizers().move_to_overshoot()
            else:
                self.eval_model = self.base_model
        
        with torch.no_grad():
            output = self.eval_model.forward(**batch)
        self.val_losses.append(output["loss"].item())
        if self.val_dataset.is_classification():
            self.val_accuracy.append(100 * torch.mean(output["logits"].argmax(dim=-1) == batch["labels"], dtype=float).item())
        
    def on_validation_epoch_end(self):
        self.val_stats.append({"epoch": self.current_epoch, "loss": float(np.mean(self.val_losses))})
        if self.val_dataset.is_classification():
            self.val_stats[-1]["accuracy"] = float(np.mean(self.val_accuracy))
        k_v_to_str = lambda k, v: f'{k}: {round(v, 4) if type(v) == float else v}'
        print(f"===TEST=== " + ' | '.join([k_v_to_str(k, v) for k, v in self.val_stats[-1].items()]), flush=True)
        self.log_dict({f"val_{k}": v for k, v in self.val_stats[-1].items()})
        self.val_losses, self.val_accuracy, self.eval_model = [], [], None

    def configure_optimizers(self):
        optimizers = []
        model_names = ["base"] if self.automatic_optimization else ["base", "overshoot"]
        for model_name in model_names:
            param_dict = {pn: p for pn, p in getattr(self, f"{model_name}_model").named_parameters() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            lr = self.config.lr * (args.overshoot_factor + 1) if model_name == "overshoot" else self.config.lr
            opt_map = {
                "sgd_momentum": torch.optim.SGD,
                "sgd_nesterov": torch.optim.SGD,
                "sgd_overshoot": SGDO,
                "sgd_adaptive": SGDO_adaptive,
                "adam": torch.optim.Adam,
                "adamW": torch.optim.AdamW,
                "adam_zero": torch.optim.Adam,
                "adamW_zero": torch.optim.AdamW,
                "nadam": torch.optim.NAdam,
                "adamW_overshoot_replication": OvershootAdamW_replication,
                "adamW_overshoot_full_approximation": OvershootAdamW_full_approximation,
                "adamW_overshoot_denom_approximation": OvershootAdamW_denom_approximation,
                "adamW_overshoot_delayed": OvershootAdamW_delayed,
                "adamW_overshoot_adaptive": OvershootAdamW_adaptive,
                "rmsprop": torch.optim.RMSprop,
            }
            if args.opt_name == "nadam":
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=lr,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    momentum_decay=1000000000000000000000000, # Turn of momentum decay
                    foreach=False,
                )
            elif args.opt_name == "adamW_overshoot_delayed":
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=lr,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    weight_decay=self.config.weight_decay,
                    overshoot=args.overshoot_factor,
                    overshoot_delay=self.config.overshoot_delay,
                    foreach=False,
                )
            elif args.opt_name == "adamW_overshoot_adaptive":
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=lr,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    weight_decay=self.config.weight_decay,
                    cosine_target=self.config.target_cosine_similarity,
                    foreach=False,
                )
            elif args.opt_name.startswith("adamW_overshoot"):
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=lr,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    weight_decay=self.config.weight_decay,
                    overshoot=args.overshoot_factor,
                    foreach=False,
                )
            elif "adam" in args.opt_name:
                self.config.adam_beta1 *= "zero" not in args.opt_name
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=lr,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    weight_decay=self.config.weight_decay,
                    foreach=False,
                )
            elif "sgd_adaptive" in args.opt_name:
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=lr,
                    momentum=self.config.sgd_momentum,
                    cosine_target=self.config.target_cosine_similarity,
                    foreach=False,
                )
            elif "sgd_overshoot" in args.opt_name:
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=lr,
                    momentum=self.config.sgd_momentum,
                    overshoot=args.overshoot_factor,
                    foreach=False,
                )
            elif "sgd" in args.opt_name:
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=lr,
                    momentum=0 if args.opt_name == "sgd" else self.config.sgd_momentum,
                    nesterov="nesterov" in args.opt_name,
                    foreach=True,
                )
            else:
                raise Exception(f"Optimizer {args.opt_name} not recognized.")

            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=self.steps)
            optimizers.append(opt)
            setattr(self, f"{model_name}_scheduler", lr_scheduler)

        return optimizers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.B, collate_fn=self.train_dataset.get_batching_fn())

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(self.val_dataset, batch_size=self.config.B, collate_fn=self.val_dataset.get_batching_fn())

    def on_train_end(self):
        dst_dir = os.path.join("lightning_logs", args.experiment_name, args.job_name)
        version = f"version_{len(next(os.walk(dst_dir))[1]) - 1}"
        pd.DataFrame(self.train_stats).to_csv(os.path.join(dst_dir, version, "training_stats.csv"), index=False)
        pd.DataFrame(self.val_stats).to_csv(os.path.join(dst_dir, version, "validation_stats.csv"), index=False)
        
# -----------------------------------------------------------------------------
def main():
    
    # 1) Create config
    trainer_config = get_trainer_config(args.model, args.dataset, args.opt_name, args.high_precision, args.config_override)
    print("-------------------------------")
    print(f"Config: {trainer_config}")
    
    # 2) Create datatset
    dataset = init_dataset(args.dataset, args.model)
    
    # 3) Create model
    model = init_model(args.model, dataset[0], trainer_config)
        # Doesn't work inside devana slurn job
        # model = torch.compile(model)
    print("-------------------------------")
    print(f"Model: {model}")


    # 4) Launch trainer
    trainer = OvershootTrainer(model, dataset, trainer_config)
    pl_trainer_args = argparse.Namespace(
        max_epochs=trainer_config.epochs,
        enable_progress_bar=False,
        enable_checkpointing=False,
        log_every_n_steps=trainer_config.log_every_n_steps,
        accumulate_grad_batches=1 if args.two_models else trainer_config.accumulate_grad_batches,
        logger=TensorBoardLogger(save_dir=os.path.join("lightning_logs", args.experiment_name), name=args.job_name),
        precision="16-mixed" if trainer_config.use_16_bit_precision else None,
        deterministic=True if args.seed else None,
        devices=trainer_config.n_gpu if trainer_config.n_gpu > 1 else "auto",
        strategy="ddp" if trainer_config.n_gpu > 1 else "auto",
    )
    print("Starting training")
    pl.Trainer(**vars(pl_trainer_args)).fit(trainer)


if __name__ == "__main__":
    # We should always observe the same results from:
    #   1) python train.py --high_precision --seed 1
    #   2) python train.py --high_precision --seed 1 --two_models --overshoot_factor 0
    # Sadly deterministic have to use 32-bit precision because of bug in pl.

    # We should observe the same results for:
    #  1)  python train.py --high_precision --model mlp --dataset mnist --seed 1 --opt_name sgd_nesterov --config_override max_steps=160
    #  2)  python train.py --high_precision --model mlp --dataset mnist --seed 1 --opt_name sgd_momentum --two_models --overshoot_factor 0.9 --config_override max_steps=160
    #  3)  python train.py --high_precision --model mlp --dataset mnist --seed 1 --opt_name sgd_overshoot --overshoot_factor 0.9 --config_override max_steps=160
    # ADD 1: In case of nesterov only overshoot model is expected to be equal

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="test", help="Folder name to store experiment results")
    parser.add_argument("--job_name", type=str, default="test", help="Sub-folder name to store experiment results")
    parser.add_argument("--overshoot_factor", type=float, help="Look-ahead factor when computng gradients")
    parser.add_argument("--two_models", action=argparse.BooleanOptionalAction, default=False, help="Use process with base and overshoot models")
    parser.add_argument("--seed", type=int, required=False, help="If specified, use this seed for reproducibility.")
    parser.add_argument("--opt_name", type=str, required=True)
    parser.add_argument("--compute_model_distance", action=argparse.BooleanOptionalAction, required=False)
    parser.add_argument("--high_precision", action=argparse.BooleanOptionalAction, required=False)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Supported types are: `gpt`, `cnn`, `gpt_hf`, `roberta_hf`, `bloom_hf`, `mdeberta_hf` and `t5_hf`. For fast iteration use `cnn`.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="""Dataset to use. Options: 
                                   a) vision: `mnist`, `cifar10`, `cifar100`
                                   b) next-token-prediction: `shakespear`, `gutenberg`,
                                   c) text-classification: `qqp`, `mnli`, `sst`""",
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
        pl.seed_everything(args.seed)
    if args.high_precision:
        torch.set_default_dtype(torch.float64)
    main()
