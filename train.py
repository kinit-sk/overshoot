import argparse
import copy
import os
import time
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForPreTraining

from cnn import CNN, ResNet50
from custom_datasets import (MnistDataset, Cifar10Dataset, Cifar100Dataset, MMLUDataset, MNLIDataset,
                             NextTokenDataloader, QQPDataset)
from custom_optimizers_rmsprop import RMSprop as CustomRMSprop
from custom_optimizers_sgd import SGD as OvershootSGD
from custom_optimizers_adamw_overshoot import AdamW as OvershootAdamW
from gpt import GPT, GPTConfig
from trainer_configs import *

# ------------------------------------------------------------------------------
pl.seed_everything(1337)
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
        self.dataset = dataset
        self.steps = int(round(config.B + config.epochs * len(dataset) // config.B // max(1, config.n_gpu)))
        self.config = config
        self.start_time = time.time()
        self.training_stats = []
        self.previous_grads = None
        self.previous_params = None
        self.last_update = None
        self.grad_cosine_sim = 0
        self.update_cosine_sim = 0

    def _baseline_training_step(self, batch):
        output = self.base_model.forward(**batch)
        self.base_scheduler.step()  # For some reason this needs to be called manually
        return output["loss"], output["loss"], output["logits"]

    def _overshoot_training_step(self, batch, batch_idx):
        if batch_idx == 0: # Most likely this is not needed
            for opt in self.optimizers():
                opt.zero_grad()

        output_overshoot = self.overshoot_model.forward(**batch)
        with torch.no_grad():
            output_base = self.base_model.forward(**batch) # only to log base loss
        output_overshoot["loss"] /= self.config.accumulate_grad_batches
        self.manual_backward(output_overshoot["loss"])

        if (batch_idx + 1) % self.config.accumulate_grad_batches == 0:
            
            # 1) Gradients OVERSHOOT -> BASE
            for (name1, param1), (name2, param2) in zip(
                self.overshoot_model.named_parameters(), self.base_model.named_parameters()
            ):
                if param1.grad is not None:
                    assert name1 == name2, "Parameter names do not match between models."
                    param2.grad = param1.grad.clone()

            if args.compute_cosine:
                grads = torch.cat([p.grad.view(-1) for _, p in self.overshoot_model.named_parameters() if p.grad is not None])
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

            # 3) Update models based on gradients
            self.base_scheduler.step(); self.overshoot_scheduler.step()
            for opt in self.optimizers():
                opt.step()
                opt.zero_grad()

        return output_base["loss"], output_overshoot["loss"], output_base["logits"]

    def training_step(self, batch, batch_idx):
        # self.trainer.should_stop = batch_idx > 300
        if self.automatic_optimization:
            loss_base, loss_overshoot, output_base = self._baseline_training_step(batch)
        else:
            loss_base, loss_overshoot, output_base = self._overshoot_training_step(batch, batch_idx)

        for device_id in range(self.config.n_gpu):
            torch.cuda.synchronize(device_id)  # wait for the GPUs to finish work

        now = time.time()
        dt = now - self.start_time  # time difference in seconds
        self.start_time = now
        
        if args.dataset in ['shakespear', 'gutenberg'] and 'gpt' in args.model:
            accuracy = 100 * torch.mean(output_base.argmax(dim=-1)[:,:-1] == batch["labels"][:,1:], dtype=float).item()
        else:
            accuracy = 100 * torch.mean(output_base.argmax(dim=-1) == batch["labels"], dtype=float).item()
            
        lr_base = self.base_scheduler.get_last_lr()[-1]
        if hasattr(self, "overshoot_scheduler"):
            lr_overshoot = self.overshoot_scheduler.get_last_lr()[-1]
        else:
            lr_overshoot = lr_base

        if batch_idx % 10 == 0:
            gpu_info = ""
            for gpu_index in range(self.config.n_gpu):
                max_vram = torch.cuda.memory_reserved(gpu_index) / (1024 * 1024 * 1024)
                utilization = torch.cuda.utilization(gpu_index)
                gpu_info += f" | vram{gpu_index} {max_vram:.2f}GB | util{gpu_index} {utilization:.2f}%"
            print(
                f"epoch: {self.current_epoch} | step {batch_idx:4d} | lr_base: {lr_base:.4f} | lr_overshoot: {lr_overshoot:.4f} | loss_base: {loss_base.item():.6f} | loss_overshoot: {loss_overshoot.item():.6f} | grad_cosine_sim: {self.grad_cosine_sim:.5f} | update_cosine_sim: {self.update_cosine_sim:.5f} | accuracy: {accuracy:.2f} | dt: {dt*1000:.2f}ms{gpu_info}", flush=True
            )

        if (batch_idx + 1) % self.config.accumulate_grad_batches == 0:
            stats = {
                "step": len(self.training_stats),
                "base_lr": lr_base,
                "overshoot_lr": lr_overshoot,
                "base_loss": loss_base.item(),
                "overshoot_loss": loss_overshoot.item(),
                "grads_cosine_similarity": self.grad_cosine_sim,
                "update_cosine_similarity": self.update_cosine_sim,
                "accuracy": accuracy,
            }
            self.training_stats.append(stats)
            self.log_dict(stats)
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
                "rmsprop_custom": CustomRMSprop, # RMSprop with bias correction term. Equivalent to Adam with beta1=0
                "sgd": torch.optim.SGD,
                "sgd_momentum": torch.optim.SGD,
                "sgd_nesterov": torch.optim.SGD,
                "sgd_overshoot": OvershootSGD,
                "adamW_overshoot": OvershootAdamW,
            }
            if args.opt_name == "nadam":
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=getattr(self.config, f"lr_{model_name}"),
                    betas=self.config.adam_betas,
                    momentum_decay=0
                )
            elif "adam" in args.opt_name:
                if "zero" in args.opt_name:
                    self.config.adam_betas = 0, self.config.adam_betas[1]
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=getattr(self.config, f"lr_{model_name}"),
                    betas=self.config.adam_betas,
                )
            elif args.opt_name == "sgd_overshoot":
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=getattr(self.config, f"lr_{model_name}"),
                    momentum=self.config.sgd_momentum,
                    overshoot=args.overshoot_factor - 1,
                )
            elif "sgd" in args.opt_name:
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=getattr(self.config, f"lr_{model_name}"),
                    momentum=0 if args.opt_name == "sgd" else self.config.sgd_momentum,
                    nesterov="nesterov" in args.opt_name,
                )
            else:
                opt = opt_map[args.opt_name](
                    optim_groups,
                    lr=getattr(self.config, f"lr_{model_name}"),
                    alpha=self.config.adam_betas[1],
                )
                
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=self.steps)
            optimizers.append(opt)
            setattr(self, f"{model_name}_scheduler", lr_scheduler)

        return optimizers

    def train_dataloader(self):
        print("Total Steps: ", self.steps)
        return DataLoader(self.dataset, batch_size=self.config.B)


def init_model(model_name, dataset_name):
    model_map = {
        "gpt_hf": "openai-community/gpt2",
        "roberta_hf": "FacebookAI/roberta-base",
        "xlm_roberta_hf": "FacebookAI/xlm-roberta-base",
        "bloom_hf": "bigscience/bloom-560m",
        "mdeberta_hf": "microsoft/mdeberta-v3-base",
        "t5_hf": "google-t5/t5-base",
    }
    dataset_to_shape = {
        "mnist": ((28, 28, 3), 10),
        "cifar10": ((32, 32, 3), 10),
        "cifar100": ((32, 32, 3), 100)
    }
    
    if model_name == "gpt":
        tokenizer = AutoTokenizer.from_pretrained(model_map["gpt_hf"]) # use tokenizer from HF
        tokenizer.pad_token = tokenizer.eos_token
        return GPT(GPTConfig(vocab_size=50304)), tokenizer
    elif model_name == "cnn":
        return CNN(dataset_to_shape[dataset_name][0], dataset_to_shape[dataset_name][1]), None
    elif model_name == "resnet50":
        return ResNet50(dataset_to_shape[dataset_name][1]), None
    elif model_name in model_map:
        model_name = model_map[model_name]
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # config.hidden_dropout_prob = 0.0  # Default is 0.1
        # config.attention_probs_dropout_prob = 0.0  # Default is 0.1
        config.ignore_mismatched_sizes = True

        if dataset_name in ['shakespear', 'gutenberg']:
            # model = AutoModelForPreTraining.from_pretrained(model_name, config=config) # pre-trained model
            model = AutoModelForPreTraining.from_config(config) # from scratch
        else:
            config.num_labels = 2
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
            
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
        model.train()
        
        return model, tokenizer
    

def init_dataset(dataset_name, tokenizer: Optional = None, T: Optional = None):
    if dataset_name == "mnist":
        return MnistDataset()
    elif dataset_name == "cifar10":
        return Cifar10Dataset()
    elif dataset_name == "cifar100":
        return Cifar100Dataset()
    elif dataset_name == "shakespear":
        return NextTokenDataloader(tokenizer, T=T, source_file="tiny_shakespear_")
        # return NextTokenDataloader(tokenizer, T=T, source_file="tiny_shakespear.txt")
    elif dataset_name == "gutenberg":
        # return NextTokenDataloader(tokenizer, T=T, source_file="gutenberg_books.txt")
        return NextTokenDataloader(tokenizer, T=T, source_file="gutenberg_books_")
    elif dataset_name == "qqp":
        return QQPDataset(tokenizer=tokenizer)
    elif dataset_name == "mnli":
        return MNLIDataset(tokenizer=tokenizer)
    elif dataset_name == "mmlu":
        return MMLUDataset(tokenizer=tokenizer)



# -----------------------------------------------------------------------------
def main():
    model, tokenizer = init_model(args.model, args.dataset)
    dataset = init_dataset(args.dataset, tokenizer, 512 if args.model in ["xlm_roberta_hf", "roberta_hf", "bert_hf"] else 1024)
    trainer_config = TrainerConfig()
    print(model, flush=True)
    
    # Doesn't work inside devana slurn job
    # model = torch.compile(model)

    sub_name = "baseline" if args.baseline else f"overshoot_factor_{args.overshoot_factor:.2f}"
    if not args.baseline:
        trainer_config.lr_overshoot = trainer_config.lr_base * args.overshoot_factor
    if args.adaptive_adam_beta and args.overshoot_factor and args.overshoot_factor > 1:
        beta1 = 1 - 1 / (2 * (args.overshoot_factor - 1))
        trainer_config.adam_betas = beta1, trainer_config.adam_betas[1]
        print(f"Using adam beta1={beta1}.")

    trainer = OvershootTrainer(model, dataset, trainer_config)
    pl_trainer_args = argparse.Namespace(
        max_epochs=trainer_config.epochs,
        enable_progress_bar=False,
        log_every_n_steps=1,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches if args.baseline else 1,
        logger=TensorBoardLogger(save_dir=os.path.join("lightning_logs", args.job_name), name=sub_name),
        devices=trainer_config.n_gpu if trainer_config.n_gpu > 1 else "auto",
        strategy="deepspeed_stage_2" if trainer_config.n_gpu > 1 else "auto",
    )
    if args.deterministic:
        print("Deterministic run which is slower.")
        pl_trainer_args.deterministic = True
    else:
        pl_trainer_args.precision = "16-mixed"
    pl.Trainer(**vars(pl_trainer_args)).fit(trainer)
    # pl_trainer.fit(trainer)
    pd.DataFrame(trainer.training_stats).to_csv(
        os.path.join("lightning_logs", args.job_name, f"{sub_name}.csv"), index=False
    )


if __name__ == "__main__":
    # We should always observe the same results from:
    #   1) python train.py --baseline --deterministic
    #   2) python train.py --overshoot_factor 1 --deterministic
    # Sadly deterministic have to use 32-bit precision because of bug in pl.

    # We should observe the same results for:
    #  1)  python train.py --model cnn --dataset mnist --deterministic --opt_name sgd_nesterov --baseline
    #  2)  python train.py --model cnn --dataset mnist --deterministic --opt_name sgd_overshoot --baseline --overshoot_factor 1.9
    #  3)  python train.py --model cnn --dataset mnist --deterministic --opt_name sgd_momentum --overshoot_factor 1.9
    # For sanity check always use accelerator='cpu' !!!

    parser = argparse.ArgumentParser()
    parser.add_argument("--job_name", type=str, default="test", help="Sub-folder name to store experiment results")
    parser.add_argument("--overshoot_factor", type=float, help="Factor to multiply base lr")
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
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Necessary to get the same results for `--baseline` and `--overshoot_factor 1`",
    )
    parser.add_argument(
        "--compute_cosine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute cosine similarity between successive vectors.",
    )
    parser.add_argument(
        "--adaptive_adam_beta",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--opt_name",
        type=str,
        default='adam',
    )
    args = parser.parse_args()
    assert (
        args.overshoot_factor or args.baseline
    ), "Overshoot factor or baseline needs to be set. See python train.py --help"
    if args.adaptive_adam_beta and ((not args.overshoot_factor) or args.overshoot_factor <=1):
        print("Warning: Adaptive adam beta only works with overshoot factor > 1.", flush=True)
    main()
