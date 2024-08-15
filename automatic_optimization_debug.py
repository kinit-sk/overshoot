
import os
import time
from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import numpy as np

from custom_datasets import NextTokenDataloader

from cnn import CNN, CNNTrainerConfig, RobertaTrainerConfig
from custom_datasets import NextTokenDataloader, Cifar100Dataset, SST2Datatset, QQPDataset, MMLUDataset, MNLIDataset
from gpt import GPT, GPTConfig, GPTTrainerConfig

# ------------------------------------------------------------------------------
pl.seed_everything(1337)
torch.cuda.empty_cache()
torch.set_float32_matmul_precision("high")
# -----------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    n_gpu: int = torch.cuda.device_count() # Use all available gpus
    B: int = 16
    T: int = 1024
    lr: float = 3e-4
    epochs: int = 4
    weight_decay: float = 0.1
    accumulate_grad_batches: int = 2
    gradient_clip_val: float = 0.5


class DebugTrainer(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, dataset, config):
        super(DebugTrainer, self).__init__()
        self.model = model
        self.dataset = dataset
        self.steps = int(round(config.B + config.epochs * len(dataset) // config.B // config.n_gpu))
        self.config = config
        
        # TODO: True/False gives differente results. Why?
        self.automatic_optimization = False


    def training_step(self, batch, batch_idx):
        if self.automatic_optimization:
            output = self.model.forward(**batch)
            loss = output['loss']
            self.lr_scheduler.step()
        else:
            opt = self.optimizers()
            opt.zero_grad()
            output = self.model.forward(**batch)
            loss = output['loss']
            self.manual_backward(loss)
            self.lr_scheduler.step()
            opt.step()
        
        
        for device_id in range(self.config.n_gpu):
            torch.cuda.synchronize(device_id)  # wait for the GPUs to finish work

        lr = self.lr_scheduler.get_last_lr()[-1]
        if batch_idx % 10 == 0:
            print(f"step {batch_idx:4d} | lr: {lr:.11f} | loss: {loss.item():.11f}")
        self.log_dict({"train_loss": loss.item(), "lr": lr})
        return loss

    def configure_optimizers(self):
        if hasattr(self, 'opt'):
            return self.opt
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
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
        self.opt = torch.optim.AdamW(optim_groups, lr=self.config.lr, betas=(0.9, 0.95), eps=1e-8, fused=False)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt, T_0=self.steps)
        return self.opt

    def train_dataloader(self):
        print("Total Steps: ", self.steps)
        return DataLoader(self.dataset, batch_size=self.config.B, shuffle=False)


# -----------------------------------------------------------------------------
def main():
    # model = GPT(GPTConfig(vocab_size=50304))
    # dataset = NextTokenDataloader(T=model.config.T, source_file='tiny_shakespear.txt')
    # trainer_config = GPTTrainerConfig()
    
    model = CNN()
    dataset = Cifar100Dataset()
    trainer_config = CNNTrainerConfig()

    # Doesn't work inside devana slurn job
    # model = torch.compile(model)

    trainer_config = TrainerConfig()
    gpt_trainer = DebugTrainer(model, dataset, trainer_config)
    trainer = pl.Trainer(
        max_epochs=trainer_config.epochs,
        # accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        # gradient_clip_val=trainer_config.gradient_clip_val,
        precision="16-mixed",
        enable_progress_bar=False,
        log_every_n_steps=1,
        deterministic=True,
        logger=TensorBoardLogger(save_dir="lightning_logs", name="demo"),
        devices=trainer_config.n_gpu,
        strategy='deepspeed_stage_2' if trainer_config.n_gpu > 1 else 'auto',
    )
    # trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False
    trainer.fit(gpt_trainer)

if __name__ == "__main__":
    main()
