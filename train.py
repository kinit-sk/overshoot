import argparse
import copy
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from cnn import CNN, CNNTrainerConfig, RobertaTrainerConfig
from custom_datasets import NextTokenDataloader, Cifar100Dataset, SST2Datatset, QQPDataset, MMLUDataset, MNLIDataset
from gpt import GPT, GPTConfig, GPTTrainerConfig

# ------------------------------------------------------------------------------
pl.seed_everything(1337)
torch.cuda.empty_cache()
torch.set_float32_matmul_precision("high")
# -----------------------------------------------------------------------------


class OvershootTrainer(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, dataset, config):
        super(OvershootTrainer, self).__init__()
        self.automatic_optimization = args.baseline
        self.base_model = model
        if not args.baseline:
            self.overshoot_model = copy.deepcopy(model)
        self.dataset = dataset
        self.steps = int(round(config.B + config.epochs * len(dataset) // config.B // config.n_gpu))
        self.config = config
        self.start_time = time.time()
        self.training_stats = []

    def _baseline_training_step(self, batch):
        output = self.base_model.forward(**batch)
        self.base_scheduler.step()  # For some reason this needs to be called manually
        return output['loss'], output['loss'], output['logits']

    def _overshoot_training_step(self, batch):
        for opt in self.optimizers():
            opt.zero_grad()

        output_overshoot = self.overshoot_model.forward(**batch)

        # Only to log base loss
        with torch.no_grad():
            output_base = self.base_model.forward(**batch)

        self.manual_backward(output_overshoot['loss'])

        # Gradients OVERSHOOT -> BASE
        for (name1, param1), (name2, param2) in zip(
            self.overshoot_model.named_parameters(), self.base_model.named_parameters()
        ):
            if param1.grad is not None:
                assert name1 == name2, "Parameter names do not match between models."
                param2.grad = param1.grad.clone()

        # Weights BASE -> OVERSHOOT
        for param1, param2 in zip(self.base_model.parameters(), self.overshoot_model.parameters()):
            param2.data = param1.data.clone()


        self.base_scheduler.step()
        self.overshoot_scheduler.step()
        
        for opt in self.optimizers():
            opt.step()
            
        return output_base['loss'], output_overshoot['loss'], output_base['logits']

    def training_step(self, batch, batch_idx):
        if self.automatic_optimization:
            loss_base, loss_overshoot, output_base = self._baseline_training_step(batch)
        else:
            loss_base, loss_overshoot, output_base = self._overshoot_training_step(batch)

        for device_id in range(self.config.n_gpu):
            torch.cuda.synchronize(device_id)  # wait for the GPUs to finish work

        now = time.time()
        dt = now - self.start_time  # time difference in seconds
        self.start_time = now
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
                f"step {batch_idx:4d} | lr_base: {lr_base:.4f} | lr_overshoot: {lr_overshoot:.4f} | loss_base: {loss_base.item():.6f} | loss_overshoot: {loss_overshoot.item():.6f} | accuracy: {accuracy:2f} | dt: {dt*1000:.2f}ms{gpu_info}"
            )

        stats = {
            "step": len(self.training_stats),
            "base_lr": lr_base,
            "overshoot_lr": lr_overshoot,
            "base_loss": loss_base.item(),
            "overshoot_loss": loss_overshoot.item(),
            "accuracy": accuracy,
        }
        self.training_stats.append(stats)
        self.log_dict(stats)
        return loss_overshoot

    def configure_optimizers(self):
        optimizers = []
        model_names = ["base"] if self.automatic_optimization else ["base", "overshoot"]
        for model_name in model_names:
            # start with all of the candidate parameters (that require grad)
            param_dict = {pn: p for pn, p in self.named_parameters()}
            param_dict = {pn: p for pn, p in param_dict.items() if pn.startswith(model_name)}
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
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
            # Create AdamW optimizer and use the fused version if it is available
            opt = torch.optim.AdamW(
                optim_groups,
                lr=getattr(self.config, f"lr_{model_name}"),
                betas=self.config.adam_betas,
                eps=1e-8,
                fused=False,
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
    if args.task_type == "gpt":
        model = GPT(GPTConfig(vocab_size=50304))
        dataset = NextTokenDataloader(T=model.config.T, source_file='tiny_shakespear.txt')
        trainer_config = GPTTrainerConfig()
    elif args.task_type == "cnn":
        model = CNN()
        dataset = Cifar100Dataset()
        trainer_config = CNNTrainerConfig()
    elif args.task_type == "roberta":
        model_name = "FacebookAI/roberta-base"
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, ignore_mismatched_sizes=True)
        model.train()
        dataset = MNLIDataset(model_name)
        trainer_config = RobertaTrainerConfig()

    print(model)
    # Doesn't work inside devana slurn job
    # model = torch.compile(model)

    sub_name = "baseline" if args.baseline else f"overshoot_factor_{args.overshoot_factor:.2f}"
    if not args.baseline:
        trainer_config.lr_overshoot = trainer_config.lr_base * args.overshoot_factor

    trainer = OvershootTrainer(model, dataset, trainer_config)
    pl_trainer = pl.Trainer(
        max_epochs=trainer_config.epochs,
        # accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        # gradient_clip_val=trainer_config.gradient_clip_val,
        precision="16-mixed",
        enable_progress_bar=False,
        log_every_n_steps=1,
        logger=TensorBoardLogger(save_dir=os.path.join("lightning_logs", args.job_name), name=sub_name),
        devices=trainer_config.n_gpu,
        strategy="deepspeed_stage_2" if trainer_config.n_gpu > 1 else "auto",
    )
    # pl_trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False
    pl_trainer.fit(trainer)
    pd.DataFrame(trainer.training_stats).to_csv(
        os.path.join("lightning_logs", args.job_name, f"{sub_name}.csv"), index=False
    )


if __name__ == "__main__":
    # We should always observe the same results from:
    #   1) python train.py --job_name test --baseline
    #   2) python train.py --job_name test --overshoot_factor 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--job_name", type=str, required=True)
    parser.add_argument("--overshoot_factor", type=float)
    parser.add_argument("--task_type", type=str, default="gpt")
    parser.add_argument("--baseline", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    assert (
        args.overshoot_factor or args.baseline
    ), "Overshoot factor or baseline needs to be set. See python train.py --help"
    main()
