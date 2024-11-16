import copy
import os
import re
import time

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
        self.train_dataset, self.val_dataset, self.test_dataset = dataset
        self.steps = int(round(config.B + config.epochs * len(self.train_dataset) // config.B // max(1, config.n_gpu)))
        if config.max_steps:
            self.steps = min(self.steps, config.max_steps)
        self.config = config
        self.train_stats, self.val_stats, self.test_stats = [], [], []
        self.current_step = 0
        self.train_losses, self.train_accuracy = [], []
        self.overshoot_losses = []
        # Cosine gradient statistics
        self.previous_params, self.previous_params_est = None, None
        self.last_update, self.last_update_est = None, None
        self.update_cosine, self.update_cosine_est = [0], [0]
        print("-----------------------------------------------")
        print("Total training steps: ", self.steps)
        print(f"Epoch steps: {len(self.train_dataset) // (config.B * max(1, config.n_gpu))}")
        print("--")
        print(f"Train dataset size: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"Valid dataset size: {len(self.val_dataset)}")
        if self.test_dataset:
            print(f"Test dataset size: {len(self.test_dataset)}")
        print("-----------------------------------------------")

    def _compute_model_distance(self):
        latest_base_model = torch.cat([p.data.view(-1).cpu() for p in self.base_model.parameters()])
        if self.two_models:
            self.past_models.append(torch.cat([p.data.view(-1).cpu() for p in self.overshoot_model.parameters()]))
        else:
            self.past_models.append(latest_base_model)
            
        if len(self.past_models) > 50:
            self.past_models.pop(0)
        momentum = self.config.sgd_momentum if "sgd" in self.args.opt_name else self.config.adam_beta1
        return compute_model_distance(latest_base_model, self.past_models, momentum)
            
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
            batch['x'] = batch['x'].cuda()
            batch['labels'] = batch['labels'].cuda()
        return batch


    def _get_base_model(self):
        if len(self.optimizers) != 1 or not hasattr(self.optimizers[0], "move_to_base"):
            return self.base_model, True
        with torch.no_grad():
            self.optimizers[0].move_to_base()
            # !!! For some reason when performing the inference on base_model, training breaks
            # This deep copy is only needed when using GPT with 16-bit precision
            base_model = copy.deepcopy(self.base_model)
            self.optimizers[0].move_to_overshoot()
        return base_model, False

    # This just prints stats to console. Shouldn't be this complicated
    def __print_stats(self, stats):
        k_v_to_str = lambda k, v: f'{k}: {round(v, 4) if type(v) == float else v}'
        text = ' | '.join([k_v_to_str(k, v) for k, v in stats.items() if not re.search(r"(loss|accuracy|similarity|est)_[0-9][0-9]+$", k)])
        print(text + (get_gpu_stats(self.config.n_gpu) if self.config.log_gpu else ''), flush=True)

    def _baseline_training_step(self, batch, batch_idx):
        assert len(self.optimizers) == 1
        optimizer = self.optimizers[0]

        base_model, is_same = self._get_base_model()
        if not is_same:
            with torch.no_grad():
                base_output = base_model.forward(**batch)
            
        output = self.base_model.forward(**batch)
        output["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()
        for scheduler in self.lr_schedulers:
            scheduler.step()
            
        if is_same:
            return output["loss"], output["loss"], output["logits"]
        else:
            return base_output["loss"], output["loss"], base_output["logits"]


    def _two_models_training_step(self, batch, batch_idx):
        assert len(self.optimizers) == 2
        with torch.no_grad():
            output_base = self.base_model.forward(**batch)  # only to log base loss
        output_overshoot = self.overshoot_model.forward(**batch)
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
            for opt in self.optimizers:
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
        }
        for avg in [1, 20, 50, 100]:
            stats[f"base_loss_{avg}"] = float(np.mean(self.train_losses[-avg:]))
            stats[f"overshoot_loss_{avg}"] = float(np.mean(self.overshoot_losses[-avg:])) # For baseline same as base loss

                
        if self.train_dataset.is_classification():
            self.train_accuracy.append(100 * torch.mean(output_base.argmax(dim=-1) == batch["labels"], dtype=float).item())
            if len(self.train_accuracy) > 100:
                self.train_accuracy.pop(0)
            for avg in [1, 20, 50, 100]:
                stats[f"accuracy_{avg}"] = float(np.mean(self.train_accuracy[-avg:]))

        if self.args.compute_cosine:
            for avg in [1, 20, 50, 100]:
                stats[f"update_cosine_similarity_{avg}"] = float(np.mean(self.update_cosine[-avg:]))
                stats[f"update_cosine_similarity_est_{avg}"] = float(np.mean(self.update_cosine_est[-avg:]))
            
        if self.args.compute_model_distance:
            stats["model_distance"] = model_distance
            
        self.train_stats.append(stats)
        if self.current_step % self.config.log_every_n_steps == 0:
            self.__print_stats(stats)
            for k, v in stats.items():
                self.log_writer.add_scalar(k, v, self.current_step)    

    def log_stats(self, stats, epoch, epoch_duration, loss, accuracy):
        self.log_writer.add_scalar('test_loss', loss, epoch)    
        self.log_writer.add_scalar('test_accuracy', accuracy, epoch)    
        stats.append({"test_loss": loss, "test_accuracy": accuracy})
        print(f"=== Epoch: {epoch} | Time: {epoch_duration:.2f} | Loss: {np.mean(loss):.4f} | Accuracy: {accuracy:.2f}%")
        

    def configure_optimizers(self):
        self.optimizers, self.lr_schedulers = [], []
        params_lr = [(self.base_model.parameters(), self.config.lr)]
        if self.two_models:
            params_lr.append((self.overshoot_model.parameters(), self.config.lr * (self.args.overshoot_factor + 1)))
            
        for params, lr in params_lr:
            self.optimizers.append(create_optimizer(self.args.opt_name, params, self.args.overshoot_factor, lr, self.config))
            if self.config.decay_lr:
                self.lr_schedulers.append(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizers[-1], T_0=self.steps))

    def save_stats(self):
        pd.DataFrame(self.train_stats).to_csv(os.path.join(self.log_writer.log_dir, "training_stats.csv"), index=False)
        if self.val_dataset:
            pd.DataFrame(self.val_stats).to_csv(os.path.join(self.log_writer.log_dir, "validation_stats.csv"), index=False)
        if self.test_dataset:
            pd.DataFrame(self.test_stats).to_csv(os.path.join(self.log_writer.log_dir, "test_stats.csv"), index=False)
            

    def main(self):
        self.configure_optimizers()
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.B, num_workers=4, shuffle=True, collate_fn=self.train_dataset.get_batching_fn())
        if self.val_dataset:
            val_dataloader = DataLoader(self.val_dataset, batch_size=self.config.B, num_workers=2, shuffle=False, collate_fn=self.val_dataset.get_batching_fn())
        else:
            val_dataloader = None
        if self.test_dataset:
            test_dataloader = DataLoader(self.test_dataset, batch_size=self.config.B, num_workers=2, shuffle=False, collate_fn=self.test_dataset.get_batching_fn())
        else:
            test_dataloader = None
            
        
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            # Training
            self._set_model_mode(is_training=True)
            for batch_id, batch in enumerate(train_dataloader):
                self.training_step(self._move_batch_to_cuda(batch), epoch, batch_id)
                self.current_step += 1
                if self.current_step >= self.steps:
                    self.save_stats()
                    return

            # Validation
            self._set_model_mode(is_training=False)
            base_model, _ = self._get_base_model()
            test_loaders = [x for x in [(val_dataloader, self.val_stats), (test_dataloader, self.test_stats)] if x[0]]
            with torch.no_grad():
                for loader, stats in test_loaders:
                    correct, total, losses = 0, 0, []
                    for batch in loader:
                        batch = self._move_batch_to_cuda(batch)
                        outputs = base_model.forward(**batch)
                        _, predicted = outputs["logits"].max(1)
                        total += batch["labels"].size(0)
                        correct += predicted.eq(batch["labels"]).sum().item()
                        losses.append(outputs["loss"].item())
                    self.log_stats(stats, epoch, time.time() - start_time, np.mean(losses), 100 * correct / total)
            self.save_stats()