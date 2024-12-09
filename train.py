import copy
import os
import re
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch


from torch.nn import functional as F
from torch.utils.data import DataLoader

from optimizers import create_optimizer


from misc import get_gpu_stats, compute_model_distance

# ------------------------------------------------------------------------------
torch.cuda.empty_cache()
# -----------------------------------------------------------------------------


class OvershootTrainer:
    def __init__(self, model: torch.nn.Module, dataset, log_writer, args, config):
        self.log_writer = log_writer
        self.args = args
        self.two_models = args.two_models
        self.base_model = model
        if args.two_models:
            self.overshoot_model = copy.deepcopy(model)
        self.steps = int(round(config.B + config.epochs * len(dataset[0]) // config.B // max(1, config.n_gpu)))
        if config.max_steps:
            self.steps = min(self.steps, config.max_steps)
        self.config = config
        self.train_stats, self.val_stats, self.test_stats = [], [], []
        self.current_step = 0
        self.train_losses, self.train_accuracy = [], []
        self.overshoot_losses = []
        # Cosine gradient statistics
        if self.args.compute_cosine:
            self.previous_params, self.previous_params_est = None, None
            self.last_update, self.last_update_est = None, None
            self.update_cosine, self.update_cosine_est = [0], [0]
        if self.args.compute_model_distance:
            self.past_weights = []
            
        # Load Dataloaders
        self.val_dataloader, self.test_dataloader = None, None
        self.train_dataloader = DataLoader(dataset[0], batch_size=self.config.B, num_workers=4, shuffle=True, collate_fn=dataset[0].get_batching_fn())
        if dataset[1]:
            self.val_dataloader = DataLoader(dataset[1], batch_size=self.config.B, num_workers=2, shuffle=False, collate_fn=dataset[1].get_batching_fn())
        if dataset[2]:
            self.test_dataloader = DataLoader(dataset[2], batch_size=self.config.B, num_workers=2, shuffle=False, collate_fn=dataset[2].get_batching_fn())
        print("-----------------------------------------------")
        print("Total training steps: ", self.steps)
        print(f"Epoch steps: {len(self.train_dataloader.dataset) // (config.B * max(1, config.n_gpu))}")
        print("--")
        print(f"Train dataset size: {len(self.train_dataloader.dataset)}")
        if self.val_dataloader:
            print(f"Valid dataset size: {len(self.val_dataloader.dataset)}")
        if self.test_dataloader:
            print(f"Test dataset size: {len(self.test_dataloader.dataset)}")
        print("-----------------------------------------------")

    def _compute_model_distance(self):
            
        latest_base_model, _ = self._get_base_model()
        latest_base_weights = torch.cat([p.data.view(-1).cpu() for p in latest_base_model.parameters()])
        
        if self.two_models:
            self.past_weights.append(torch.cat([p.data.view(-1).cpu() for p in self.overshoot_model.parameters()]))
        else:
            self.past_weights.append(torch.cat([p.data.view(-1).cpu() for p in self.base_model.parameters()]))
            
        if len(self.past_weights) > 50:
            self.past_weights.pop(0)
            
        decay_factor = self.config.sgd_momentum if "sgd" in self.args.opt_name else self.config.adam_beta1
        if (self.current_step + 1) % 50 == 0:
            return compute_model_distance(latest_base_weights, self.past_weights, decay_factor)
        else:
            return -1
            
    def _cosine_similarity(self, sample_size: int = 1000):
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


    def _set_model_mode(self, is_training: bool):
        if is_training:
            self.base_model.train()
            if self.two_models:
                self.overshoot_model.train()
        else:
            self.base_model.eval()
            if self.two_models:
                self.overshoot_model.eval()
                
    
    def _move_batch_to_cuda(self, batch):
        if self.config.n_gpu > 0:
            for k in batch.keys():
                batch[k] = batch[k].cuda()
        return batch


    def _get_base_model(self):
        assert self.two_models == (len(self.optimizers) > 1)
        if self.two_models or not hasattr(self.optimizers[0], "move_to_base") or not hasattr(self.optimizers[0], "move_to_overshoot"):
            return self.base_model, True
        self.optimizers[0].move_to_base()
        # !!! For some reason when performing the inference on base_model, training breaks
        # This deep copy is only needed when using GPT with 16-bit precision
        # TODO: Figure out why this deep copy is needed (maybe inference is not read only?)
        base_model = copy.deepcopy(self.base_model)
        self.optimizers[0].move_to_overshoot()
        return base_model, False

    # This just prints stats to console. Shouldn't be this complicated
    def __print_stats(self, stats):
        k_v_to_str = lambda k, v: f'{k}: {round(v, 4) if type(v) == float else v}'
        text = ' | '.join([k_v_to_str(k, v) for k, v in stats.items() if not re.search(r"(loss|accuracy|similarity|est)_[0-9][0-9]+$", k)])
        print(text + (get_gpu_stats(self.config.n_gpu) if self.config.log_gpu else ''), flush=True)

    def model_forward_(self, model, batch):
        if self.config.n_gpu and self.config.precision == "16-mixed":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model.forward(**batch)
        return model.forward(**batch)

    def _baseline_training_step(self, batch, batch_idx):
        assert len(self.optimizers) == 1
        optimizer = self.optimizers[0]
        
        if self.args.compute_base_model_loss:
            base_model, is_same = self._get_base_model()
            if not is_same:
                with torch.no_grad():
                    base_output = self.model_forward_(base_model, batch)
                    
        output = self.model_forward_(self.base_model, batch)
        if len(self.scalers) == 1:
            self.scalers[0].scale(output["loss"]).backward()
            self.scalers[0].step(optimizer)
            self.scalers[0].update()
        else:
            output["loss"].backward()
            optimizer.step()
        
        
        optimizer.zero_grad()
        for scheduler in self.lr_schedulers:
            scheduler.step()
            
        if self.args.compute_base_model_loss and not is_same:
            return base_output["loss"], output["loss"], base_output["logits"]
        else:
            return output["loss"], output["loss"], output["logits"]


    def _two_models_training_step(self, batch, batch_idx):
        assert len(self.optimizers) == 2 and (len(self.scalers) == 0 or len(self.scalers) == 2) 
        with torch.no_grad():
            output_base = self.model_forward_(self.base_model, batch) # only to log base loss
            
        output_overshoot = self.model_forward_(self.overshoot_model, batch)
        if len(self.scalers):
            self.scalers[1].scale(output_overshoot["loss"]).backward()
        else:
            output_overshoot["loss"].backward()
        # self.manual_backward(output_overshoot["loss"] / self.config.accumulate_grad_batches)

        if (batch_idx + 1) % self.config.accumulate_grad_batches == 0:

            # 1) Gradients OVERSHOOT -> BASE
            for param1, param2 in zip(self.overshoot_model.parameters(), self.base_model.parameters()):
                if param1.grad is not None:
                    param2.grad = param1.grad.clone()

            # 2) Weights BASE -> OVERSHOOT
            for param1, param2 in zip(self.base_model.parameters(), self.overshoot_model.parameters()):
                param2.data = param1.data.clone()

            # 3) (Optional) Update learning rates
            for scheduler in self.lr_schedulers:
                scheduler.step()

            # 4) Update models based on gradients
            for i, opt in enumerate(self.optimizers):
                if len(self.scalers):
                    self.scalers[i].step(opt)
                    self.scalers[i].update()
                else:
                    opt.step()
                opt.zero_grad()

        return output_base["loss"], output_overshoot["loss"], output_base["logits"]

    def training_step(self, batch, epoch, batch_idx):
        # We compute model distances before model update to have the same behaviour for baseline and overshoot
        if self.args.compute_model_distance:
            model_distance = self._compute_model_distance()
            
        if self.args.compute_cosine:
            self._cosine_similarity()

        train_fn = self._two_models_training_step if self.two_models else self._baseline_training_step 
        loss_base, loss_overshoot, output_base = train_fn(batch, batch_idx)

        for losses, new_loss in [(self.train_losses, loss_base.item()), (self.overshoot_losses, loss_overshoot.item())]:
            losses.append(new_loss)
            if len(losses) > 100:
                losses.pop(0)
            
        stats = {
            "step": self.current_step,
            "epoch": epoch,
            "batch_step": batch_idx,
            "wall_time": time.time() - self.trainig_start_time,
        }
        for avg in [1, 20, 50, 100]:
            stats[f"base_loss_{avg}"] = float(np.mean(self.train_losses[-avg:]))
            stats[f"overshoot_loss_{avg}"] = float(np.mean(self.overshoot_losses[-avg:])) # For baseline same as base loss

                
        if self.train_dataloader.dataset.is_classification():
            self.train_accuracy.append(100 * torch.mean(output_base.argmax(dim=-1) == batch["labels"], dtype=float).item())
            if len(self.train_accuracy) > 100:
                self.train_accuracy.pop(0)
            for avg in [1, 20, 50, 100]:
                stats[f"accuracy_{avg}"] = float(np.mean(self.train_accuracy[-avg:]))

        for i, scheduler in enumerate(self.lr_schedulers):
            stats[f"lr_{i}"] = scheduler.get_last_lr()[-1]

        if self.args.compute_cosine:
            for avg in [1, 20, 50, 100]:
                stats[f"update_cosine_similarity_{avg}"] = float(np.mean(self.update_cosine[-avg:]))
                stats[f"update_cosine_similarity_est_{avg}"] = float(np.mean(self.update_cosine_est[-avg:]))
            
        if self.args.compute_model_distance:
            stats["model_distance"] = model_distance

        if hasattr(self.optimizers[0], "_overshoot_new"):
            stats["adaptive_overshoot"] = self.optimizers[0]._overshoot_new
            
        self.train_stats.append(stats)
        if self.current_step % self.config.log_every_n_steps == 0:
            self.__print_stats(stats)
            for k, v in stats.items():
                self.log_writer.add_scalar(k, v, self.current_step)    

    def log_stats(self, name, stats, epoch, loss, accuracy):
        now = time.time()
        wall_time, epoch_duration = now - self.trainig_start_time, now - self.epoch_start
        stats.append({"loss": loss, "wall_time": wall_time})
        self.log_writer.add_scalar(f'{name}_loss', loss, epoch)    
        if accuracy:
            self.log_writer.add_scalar(f'{name}_accuracy', accuracy, epoch)    
            stats[-1]["accuracy"] = accuracy
        print(f"=== Epoch: {epoch} | Wall Time: {wall_time:.2f} | Epoch Time: {epoch_duration:.2f} | Loss: {np.mean(loss):.4f}" + (f" | Accuracy: {accuracy:.2f}%" if accuracy else ""))
        

    def configure_optimizers(self):
        self.optimizers, self.scalers, self.lr_schedulers = [], [], []
        params_lr = [(self.base_model.parameters(), self.config.lr)]
        if self.two_models:
            params_lr.append((self.overshoot_model.parameters(), self.config.lr * (self.args.overshoot_factor + 1)))
            
        for params, lr in params_lr:
             # TODO: Scaler for two model is not working
            if self.config.precision == "16-mixed" and self.two_models == False:
                self.scalers.append(torch.GradScaler(device="cuda"))
            self.optimizers.append(create_optimizer(self.args.opt_name, params, self.args.overshoot_factor, lr, self.config))
            if self.config.use_lr_scheduler:
                self.lr_schedulers.append(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizers[-1], T_0=self.steps))

    def save_stats(self):
        pd.DataFrame(self.train_stats).to_csv(os.path.join(self.log_writer.log_dir, "training_stats.csv"), index=False)
        if self.val_stats:
            pd.DataFrame(self.val_stats).to_csv(os.path.join(self.log_writer.log_dir, "validation_stats.csv"), index=False)
        if self.test_stats:
            pd.DataFrame(self.test_stats).to_csv(os.path.join(self.log_writer.log_dir, "test_stats.csv"), index=False)

    def validation(self, epoch):
        self._set_model_mode(is_training=False)
        base_model, _ = self._get_base_model()
        test_loaders = [x for x in [(self.val_dataloader, self.val_stats, "validation"), (self.test_dataloader, self.test_stats, "test")] if x[0]]
        with torch.no_grad():
            for loader, stats, name in test_loaders:
                correct, total, loss = 0, 0, 0
                for batch in loader:
                    batch = self._move_batch_to_cuda(batch)
                    outputs = self.model_forward_(base_model, batch)
                    _, predicted = outputs["logits"].max(1)
                    loss += outputs["loss"].item() * batch["labels"].size(0)
                    if loader.dataset.is_classification():
                        total += batch["labels"].size(0)
                        correct += predicted.eq(batch["labels"]).sum().item()
                accuracy = 100 * correct / total if loader.dataset.is_classification() else None
                self.log_stats(name, stats, epoch, loss / len(loader.dataset), accuracy)
        self.save_stats()
            

    def main(self):
        self.configure_optimizers()
        
        self.trainig_start_time = time.time()
        for epoch in range(self.config.epochs):
            self.epoch_start = time.time()
            
            # Training
            self._set_model_mode(is_training=True)
            for batch_id, batch in enumerate(self.train_dataloader):
                self.training_step(self._move_batch_to_cuda(batch), epoch, batch_id)
                self.current_step += 1
                if self.current_step >= self.steps:
                    self.validation(epoch)
                    print("Max steps reached. Finished training.")
                    return

            self.validation(epoch)