import argparse
import copy
import os
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader

from custom_optimizers_adamw_overshoot import AdamW as OvershootAdamW
from custom_optimizers_adamw_overshoot_v2 import AdamW as OvershootAdamW_v2
from custom_optimizers_rmsprop import RMSprop as CustomRMSprop
from custom_optimizers_sgd import SGD as OvershootSGD
from misc import init_dataset, init_model
from trainer_configs import *

# ------------------------------------------------------------------------------
torch.cuda.empty_cache()
torch.set_float32_matmul_precision("high")
# -----------------------------------------------------------------------------


class OvershootTrainer(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, dataset, config):
        super(OvershootTrainer, self).__init__()
        # Manual optimization: https://lightning.ai/docs/pytorch/stable/common/optimization.html
        self.automatic_optimization = args.baseline
        self.base_model = model
        if not args.baseline:
            self.overshoot_model = copy.deepcopy(model)
            config.lr_overshoot = config.lr_base * (args.overshoot_factor + 1)
        self.dataset = dataset
        self.steps = int(round(config.B + config.epochs * len(dataset) // config.B // max(1, config.n_gpu)))
        if config.max_steps:
            self.steps = min(self.steps, config.max_steps)
        self.config = config
        self.start_time = time.time()
        self.current_step = 0
        # Cosine gradient statistics
        self.previous_grads = None
        self.previous_params = None
        self.last_update = None
        self.grad_cosine_sim = 0
        self.update_cosine_sim = 0

    def _baseline_training_step(self, batch):
        output = self.base_model.forward(**batch)
        if self.config.decay_lr:
            self.base_scheduler.step()
        return output["loss"], output["loss"], output["logits"]

    def _overshoot_training_step(self, batch, batch_idx):
        if batch_idx == 0:  # Most likely this is not needed
            for opt in self.optimizers():
                opt.zero_grad()

        output_overshoot = self.overshoot_model.forward(**batch)
        with torch.no_grad():
            output_base = self.base_model.forward(**batch)  # only to log base loss
        self.manual_backward(output_overshoot["loss"] / self.config.accumulate_grad_batches)

        if (batch_idx + 1) % self.config.accumulate_grad_batches == 0:

            # 1) Gradients OVERSHOOT -> BASE
            for (name1, param1), (name2, param2) in zip(
                self.overshoot_model.named_parameters(), self.base_model.named_parameters()
            ):
                if param1.grad is not None:
                    assert name1 == name2, "Parameter names do not match between models."
                    param2.grad = param1.grad.clone()

            if args.compute_cosine:
                grads = torch.cat(
                    [p.grad.view(-1) for _, p in self.overshoot_model.named_parameters() if p.grad is not None]
                )
                params = torch.cat([p.data.view(-1) for p in self.overshoot_model.parameters()])
                if self.previous_grads is not None:
                    sim = F.cosine_similarity(self.previous_grads, grads, dim=0)
                    if not torch.isnan(sim).item():
                        self.grad_cosine_sim = 0.9 * self.grad_cosine_sim + 0.1 * sim.item()
                if self.previous_params is not None:
                    update = params - self.previous_params
                    if self.last_update is not None:
                        sim = F.cosine_similarity(self.last_update, update, dim=0)
                        if not torch.isnan(sim).item():
                            self.update_cosine_sim = 0.9 * self.update_cosine_sim + 0.1 * sim.item()
                    self.last_update = update
                self.previous_grads = grads

            # 2) Weights BASE -> OVERSHOOT
            for param1, param2 in zip(self.base_model.parameters(), self.overshoot_model.parameters()):
                param2.data = param1.data.clone()

            self.previous_params = torch.cat([p.data.view(-1) for p in self.overshoot_model.parameters()])

            # (Optional) 3) Update learning rates
            if self.config.decay_lr:
                self.base_scheduler.step()
                self.overshoot_scheduler.step()

            # 4) Update models based on gradients
            for opt in self.optimizers():
                opt.step()
                opt.zero_grad()

        return output_base["loss"], output_overshoot["loss"], output_base["logits"]

    def training_step(self, batch, batch_idx):
        self.trainer.should_stop = self.current_step >= self.steps

        if self.automatic_optimization:
            # TODO: How to compute base loss when having only overshoot models
            # if hasattr(self.optimizers(), "move_to_base"):
            #     with torch.no_grad():
            #         self.optimizers().move_to_base()
            #         base_output_dict = self.base_model.forward(**batch)
            #         loss_base = base_output_dict["loss"]
            #         output_base = base_output_dict["logits"]
            #         self.optimizers().move_to_overshoot()
            #     _, loss_overshoot, _ = self._baseline_training_step(batch)
            # else:
            #     loss_base, loss_overshoot, output_base = self._baseline_training_step(batch)
            loss_base, loss_overshoot, output_base = self._baseline_training_step(batch)

        else:
            loss_base, loss_overshoot, output_base = self._overshoot_training_step(batch, batch_idx)

        for device_id in range(self.config.n_gpu):
            torch.cuda.synchronize(device_id)  # wait for the GPUs to finish work

        now = time.time()
        dt = now - self.start_time  # time difference in seconds
        self.start_time = now

        if args.dataset in ["shakespear", "gutenberg"] and "gpt" in args.model:
            accuracy = (
                100 * torch.mean(output_base.argmax(dim=-1)[:, :-1] == batch["labels"][:, 1:], dtype=float).item()
            )
        else:
            accuracy = 100 * torch.mean(output_base.argmax(dim=-1) == batch["labels"], dtype=float).item()

        lr_base = self.base_scheduler.get_last_lr()[-1]
        if hasattr(self, "overshoot_scheduler"):
            lr_overshoot = self.overshoot_scheduler.get_last_lr()[-1]
        else:
            lr_overshoot = lr_base

        if batch_idx % self.config.log_every_n_steps == 0:
            gpu_info = ""
            if self.config.log_gpu and self.config.n_gpu > 0:
                for gpu_index in range(self.config.n_gpu):
                    max_vram = torch.cuda.memory_reserved(gpu_index) / (1024 * 1024 * 1024)
                    utilization = torch.cuda.utilization(gpu_index)
                    gpu_info += f" | vram{gpu_index} {max_vram:.2f}GB | util{gpu_index} {utilization:.2f}%"
            print(
                f"epoch: {self.current_epoch} | step {batch_idx:4d} | lr_base: {lr_base:.4f} | lr_overshoot: {lr_overshoot:.4f} | loss_base: {loss_base.item():.6f} | loss_overshoot: {loss_overshoot.item():.6f} | grad_cosine_sim: {self.grad_cosine_sim:.5f} | update_cosine_sim: {self.update_cosine_sim:.5f} | accuracy: {accuracy:.2f} | dt: {dt*1000:.2f}ms{gpu_info}",
                flush=True,
                # f"epoch: {self.current_epoch} | step {batch_idx:4d} | lr_base: {lr_base:.4f} | loss_base: {loss_base.item():.6f} | loss_overshoot: {loss_overshoot.item():.6f}", flush=True
            )

        if (batch_idx + 1) % self.config.accumulate_grad_batches == 0:
            stats = {
                "step": self.current_step,
                "base_lr": lr_base,
                "overshoot_lr": lr_overshoot,
                "base_loss": loss_base.item(),
                "overshoot_loss": loss_overshoot.item(),
                "grads_cosine_similarity": self.grad_cosine_sim,
                "update_cosine_similarity": self.update_cosine_sim,
                "accuracy": accuracy,
            }
            self.log_dict(stats)
        self.current_step += 1
        return loss_overshoot

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
            opt_map = {
                "adam": torch.optim.Adam,
                "adamW": torch.optim.AdamW,
                "adam_zero": torch.optim.Adam,
                "adamW_zero": torch.optim.AdamW,
                "nadam": torch.optim.NAdam,
                "rmsprop": torch.optim.RMSprop,
                "rmsprop_custom": CustomRMSprop,  # RMSprop with bias correction term. Equivalent to Adam with beta1=0
                "sgd": torch.optim.SGD,
                "sgd_momentum": torch.optim.SGD,
                "sgd_nesterov": torch.optim.SGD,
                "sgd_overshoot": OvershootSGD,
                "adamW_overshoot": OvershootAdamW,
                "adamW_overshoot_v2": OvershootAdamW_v2,
            }
            if args.opt_name == "nadam":
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=getattr(self.config, f"lr_{model_name}"),
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    momentum_decay=0,
                    foreach=False,
                )
            elif args.opt_name.startswith("adamW_overshoot"):
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=getattr(self.config, f"lr_{model_name}"),
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    weight_decay=self.config.weight_decay,
                    overshoot=args.overshoot_factor,
                    foreach=True,
                )
            elif "adam" in args.opt_name:
                self.config.adam_beta1 *= "zero" not in args.opt_name
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=getattr(self.config, f"lr_{model_name}"),
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    weight_decay=self.config.weight_decay,
                    foreach=True,
                )
            elif args.opt_name == "sgd_overshoot":
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=getattr(self.config, f"lr_{model_name}"),
                    momentum=self.config.sgd_momentum,
                    overshoot=args.overshoot_factor,
                    foreach=True,
                )
            elif "sgd" in args.opt_name:
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=getattr(self.config, f"lr_{model_name}"),
                    momentum=0 if args.opt_name == "sgd" else self.config.sgd_momentum,
                    nesterov="nesterov" in args.opt_name,
                    foreach=True,
                )
            else:
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=getattr(self.config, f"lr_{model_name}"),
                    alpha=self.config.adam_beta2,
                    foreach=False,
                )

            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=self.steps)
            optimizers.append(opt)
            setattr(self, f"{model_name}_scheduler", lr_scheduler)

        return optimizers

    def train_dataloader(self):
        print("Total Steps: ", self.steps)
        return DataLoader(self.dataset, batch_size=self.config.B)


# -----------------------------------------------------------------------------
def main():
    model, tokenizer = init_model(args.model, args.dataset)
    dataset = init_dataset(
        args.dataset, tokenizer, 512 if args.model in ["xlm_roberta_hf", "roberta_hf", "bert_hf"] else 1024
    )
    trainer_config = TrainerConfig(args.config_override)
    print(f"Model: {model}")
    print(f"Config: {trainer_config}")

    # Doesn't work inside devana slurn job
    # model = torch.compile(model)

    trainer = OvershootTrainer(model, dataset, trainer_config)
    pl_trainer_args = argparse.Namespace(
        max_epochs=trainer_config.epochs,
        enable_progress_bar=False,
        log_every_n_steps=trainer_config.log_every_n_steps,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches if args.baseline else 1,
        logger=TensorBoardLogger(save_dir=os.path.join("lightning_logs", args.experiment_name), name=args.job_name),
        precision="16-mixed" if trainer_config.use_16_bit_precision else None,
        deterministic=True if args.seed else None,
        devices=trainer_config.n_gpu if trainer_config.n_gpu > 1 else "auto",
        strategy="deepspeed_stage_2" if trainer_config.n_gpu > 1 else "auto",
    )
    print("Starting training")
    pl.Trainer(**vars(pl_trainer_args)).fit(trainer)


if __name__ == "__main__":
    # We should always observe the same results from:
    #   1) python train.py --baseline --seed 1
    #   2) python train.py --overshoot_factor 0 --seed 1
    # Sadly deterministic have to use 32-bit precision because of bug in pl.

    # We should observe the same results for:
    #  1)  python train.py --model cnn --dataset mnist --seed 1 --opt_name sgd_nesterov --baseline
    #  2)  python train.py --model cnn --dataset mnist --seed 1 --opt_name sgd_overshoot --baseline --overshoot_factor 0.9
    #  3)  python train.py --model cnn --dataset mnist --seed 1 --opt_name sgd_momentum --overshoot_factor 0.9
    # For sanity check always use accelerator='cpu' !!!

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="test", help="Folder name to store experiment results")
    parser.add_argument("--job_name", type=str, default="test", help="Sub-folder name to store experiment results")
    parser.add_argument("--overshoot_factor", type=float, help="Look-ahead factor when computng gradients")
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
                                   c) text-classification: `qqp`, `mnli`, `mmlu`""",
    )
    parser.add_argument(
        "--baseline", action=argparse.BooleanOptionalAction, default=False, help="Default adam optimization process"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="If specified, use this seed for reproducibility.",
    )
    parser.add_argument(
        "--compute_cosine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute cosine similarity between successive vectors.",
    )
    parser.add_argument(
        "--opt_name",
        type=str,
        default="adam",
    )
    parser.add_argument(
        "--config_override",
        type=str,
        nargs="+",
        default=None,
        help="Sequence of key-value pairs to override config. E.g. --config_override lr_base=0.01",
    )
    args = parser.parse_args()
    assert (
        args.overshoot_factor or args.baseline
    ), "Overshoot factor or baseline needs to be set. See python train.py --help"
    if args.seed:
        pl.seed_everything(args.seed)
    main()
