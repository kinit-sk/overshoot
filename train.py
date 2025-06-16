import argparse
import copy
import os
import re
import time
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from trainer_configs import DefaultConfig
from custom_datasets import UnifiedDatasetInterface
from misc import compute_model_distance, get_gpu_stats
from misc import create_optimizer

# ------------------------------------------------------------------------------
torch.cuda.empty_cache()
# -----------------------------------------------------------------------------


class OvershootTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Tuple[UnifiedDatasetInterface, Optional[UnifiedDatasetInterface], Optional[UnifiedDatasetInterface]],
        log_writer: Any, # SummaryWriter is untyped
        args: argparse.Namespace,
        config:  DefaultConfig,
    ):
        self.config = config
        self.log_writer = log_writer
        self.args = args
        self.base_model = model
        if args.two_models:
            self.overshoot_model = copy.deepcopy(model)
        self.steps = int(round(config.B + config.epochs * len(dataset[0]) // config.B // max(1, config.n_gpu)))
        if config.max_steps:
            self.steps = min(self.steps, config.max_steps)
        self.train_stats: list[dict[str, float]] = []
        self.val_stats: list[dict[str, float]] = []
        self.test_stats: list[dict[str, float]] = []
        self.current_step = 0
        # Cosine gradient statistics
        if self.args.compute_cosine:
            self.previous_params, self.previous_params_est = torch.empty(0), torch.empty(0)
            self.last_update, self.last_update_est = torch.empty(0), torch.empty(0)
            self.update_cosine, self.update_cosine_est = [0.0], [0.0]
        if self.args.compute_model_distance:
            self.past_weights: list[torch.Tensor] = []

        # Load Dataloaders
        self.val_dataloader: Optional[DataLoader[UnifiedDatasetInterface]] = None
        self.test_dataloader: Optional[DataLoader[UnifiedDatasetInterface]] = None
        self.train_dataloader: DataLoader[UnifiedDatasetInterface] = DataLoader(
            dataset[0], batch_size=self.config.B, num_workers=4, shuffle=True, collate_fn=dataset[0].get_batching_fn()
        )
        if dataset[1]:
            self.val_dataloader = DataLoader(
                dataset[1],
                batch_size=self.config.B,
                num_workers=2,
                shuffle=False,
                collate_fn=dataset[1].get_batching_fn(),
            )
        if dataset[2]:
            self.test_dataloader = DataLoader(
                dataset[2],
                batch_size=self.config.B,
                num_workers=2,
                shuffle=False,
                collate_fn=dataset[2].get_batching_fn(),
            )
        print("-----------------------------------------------")
        print("Total training steps: ", self.steps)
        print(f"Epoch steps: {len(self.train_dataloader.dataset) // (config.B * max(1, config.n_gpu))}")  # type: ignore
        print("--")
        print(f"Train dataset size: {len(self.train_dataloader.dataset)}")  # type: ignore
        if self.val_dataloader:
            print(f"Valid dataset size: {len(self.val_dataloader.dataset)}")  # type: ignore
        if self.test_dataloader:
            print(f"Test dataset size: {len(self.test_dataloader.dataset)}")  # type: ignore
        print("-----------------------------------------------")

    def _is_update_batch(self, batch_id: int) -> bool:
        return (batch_id + 1) % self.config.accumulate_grad_batches == 0

    def _compute_model_distance(self) -> float:
        latest_base_model, _ = self._get_base_model()
        latest_base_weights = torch.cat([p.data.view(-1).cpu() for p in latest_base_model.parameters()])

        if self.args.two_models:
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

    def _cosine_similarity(self, sample_size: int = 1000) -> None:
        params = torch.cat([p.data.view(-1) for p in self.base_model.parameters()])
        if not hasattr(self, "random_indices"):
            self.random_indices = torch.randint(0, params.size(0), (sample_size,))
        params_est = params[self.random_indices]
        if self.previous_params.numel():  # non empty
            update = params - self.previous_params
            update_est = params_est - self.previous_params_est
            if self.last_update.numel():  # non empty
                similarity = F.cosine_similarity(self.last_update, update, dim=0)
                similarity_est = F.cosine_similarity(self.last_update_est, update_est, dim=0)
                for arr, new in [(self.update_cosine, similarity), (self.update_cosine_est, similarity_est)]:
                    if not torch.isnan(new):
                        arr.append(float(new.item()))
                        if len(arr) > 100:
                            arr.pop(0)
            self.last_update = update
            self.last_update_est = update_est
        self.previous_params = params
        self.previous_params_est = params_est

    def _set_model_mode(self, is_training: bool) -> None:
        if is_training:
            self.base_model.train()
            if self.args.two_models:
                self.overshoot_model.train()
        else:
            self.base_model.eval()
            if self.args.two_models:
                self.overshoot_model.eval()

    def _move_batch_to_cuda(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.config.n_gpu > 0:
            for k in batch.keys():
                batch[k] = batch[k].cuda()
        return batch

    def _get_base_model(self) -> Tuple[torch.nn.Module, bool]:
        assert self.args.two_models == (len(self.optimizers) > 1)
        if (
            self.args.two_models
            or not hasattr(self.optimizers[0], "move_to_base")
            or not hasattr(self.optimizers[0], "move_to_overshoot")
        ):
            return self.base_model, True
        self.optimizers[0].move_to_base()
        # !!! For some reason when performing the inference on base_model, training breaks
        # This deep copy is only needed when using GPT with 16-bit precision
        # Reason:
        #   a) Forward pass with `train_mode == True` updates batch-norm statistics
        #   b) AMP casts weights in-place
        base_model = copy.deepcopy(self.base_model)
        self.optimizers[0].move_to_overshoot()
        return base_model, False

    # This just prints stats to console. Shouldn't be this complicated
    def _print_stats(self, stats: dict[str, float]) -> None:
        def k_v_to_str(k: str, v: float | int) -> str:
            return f"{k}: {round(v, 4) if isinstance(v, float) else v}"
        text = " | ".join(
            [
                k_v_to_str(k, v)
                for k, v in stats.items()
                if not re.search(r"(loss|accuracy|similarity|est)_[0-9][0-9]+$", k)
            ]
        )
        print(text + (get_gpu_stats(self.config.n_gpu) if self.config.log_gpu else ""), flush=True)

    def _model_forward(self, model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> Any:
        if self.config.n_gpu and self.config.precision == "16-mixed":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model.forward(**batch)
        return model.forward(**batch)

    def _baseline_training_step(self, batch: dict[str, torch.Tensor], batch_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(self.optimizers) == 1
        optimizer = self.optimizers[0]

        if self.args.compute_base_model_loss:
            base_model, is_same = self._get_base_model()
            if not is_same:
                with torch.no_grad():
                    base_output = self._model_forward(base_model, batch)

        output = self._model_forward(self.base_model, batch)
        output["loss"] /= self.config.accumulate_grad_batches 
        
        if self.scaler:
            self.scaler.scale(output["loss"]).backward()
        else:
            output["loss"].backward()
            
        if self._is_update_batch(batch_id):
            if self.scaler:
                self.scaler.unscale_(optimizer)
                if self.config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), self.config.grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                if self.config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), self.config.grad_clip)
                optimizer.step()
                
            optimizer.zero_grad()
            for scheduler in self.lr_schedulers:
                scheduler.step()

        if self.args.compute_base_model_loss and not is_same:
            return base_output["loss"], output["loss"], base_output["logits"]
        return output["loss"], output["loss"], output["logits"]

    def _two_models_training_step(self, batch: dict[str, torch.Tensor], batch_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(self.optimizers) == 2
        with torch.no_grad():
            output_base = self._model_forward(self.base_model, batch)  # only to log base loss
        output_overshoot = self._model_forward(self.overshoot_model, batch)
        output_overshoot["loss"] /= self.config.accumulate_grad_batches 
        output_overshoot["loss"].backward()

        if self._is_update_batch(batch_id):

            # 0) (Optional) Clip gradients
            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), self.config.grad_clip)

            # 1) Gradients OVERSHOOT -> BASE
            for param1, param2 in zip(self.overshoot_model.parameters(), self.base_model.parameters()):
                if param1.grad is not None:
                    param2.grad = param1.grad.clone()

            # 2) Weights BASE -> OVERSHOOT
            for param1, param2 in zip(self.base_model.parameters(), self.overshoot_model.parameters()):
                param2.data = param1.data.clone()

            # 3) Update models based on gradients
            for opt in self.optimizers:
                opt.step()
                opt.zero_grad()
                
            # 4) (Optional) Update learning rates
            for scheduler in self.lr_schedulers:
                scheduler.step()

        return output_base["loss"], output_overshoot["loss"], output_base["logits"]

    def _training_step(self, batch: dict[str, torch.Tensor], epoch: int, batch_id: int) -> None:
        # We compute model distances before model update to have the same behaviour for baseline and overshoot
        if self.args.compute_model_distance:
            model_distance = self._compute_model_distance()

        if self.args.compute_cosine:
            self._cosine_similarity()

        train_fn = self._two_models_training_step if self.args.two_models else self._baseline_training_step
        loss_base, loss_overshoot, output_base = train_fn(batch, batch_id)

        # Record stats only for the very last batch in `accumulate_grad_batches`
        if not self._is_update_batch(batch_id):
            return

        stats = {
            "step": self.current_step,
            "epoch": epoch,
            "batch_step": batch_id,
            "wall_time": time.time() - self.training_start_time,
        }
        
        for avg in [1, 20, 50, 100]:
            for loss_name, loss_value in [("base_loss", loss_base.item()), ("overshoot_loss", loss_overshoot.item())]:
                last_n_loss = [x[f"{loss_name}_1"] for x in self.train_stats[-avg:]] + [loss_value]
                if len(last_n_loss) == avg + 1:
                    last_n_loss.pop(0)
                stats[f"{loss_name}_{avg}"] = float(np.mean(last_n_loss))

            if self.train_dataloader.dataset.is_classification():  # type: ignore
                acc = 100 * torch.mean(output_base.argmax(dim=-1) == batch["labels"], dtype=float).item()  # type: ignore
                last_n_acc = [x[f"accuracy_1"] for x in self.train_stats[-avg:]] + [acc]
                if len(last_n_acc) == avg + 1:
                    last_n_acc.pop(0)
                stats[f"accuracy_{avg}"] = float(np.mean(last_n_acc))
                
            if self.args.compute_cosine:
                stats[f"update_cosine_similarity_{avg}"] = float(np.mean(self.update_cosine[-avg:]))
                stats[f"update_cosine_similarity_est_{avg}"] = float(np.mean(self.update_cosine_est[-avg:]))

        for i, scheduler in enumerate(self.lr_schedulers):
            stats[f"lr_{i}"] = scheduler.get_last_lr()[-1]

        if self.args.compute_model_distance:
            stats["model_distance"] = model_distance

        self.train_stats.append(stats)
        if self.current_step % self.config.log_every_n_steps == 0:
            self._print_stats(stats)
            for k, v in stats.items():
                self.log_writer.add_scalar(k, v, self.current_step)

    def log_stats(self, name: str, stats: list[dict[str, float]], epoch: int, loss: float, accuracy: Optional[float]) -> None:
        now = time.time()
        wall_time, epoch_duration = now - self.training_start_time, now - self.epoch_start
        stats.append({"loss": loss, "wall_time": wall_time})
        self.log_writer.add_scalar(f"{name}_loss", loss, epoch)
        if accuracy:
            self.log_writer.add_scalar(f"{name}_accuracy", accuracy, epoch)
            stats[-1]["accuracy"] = accuracy
        print(
            f"===[{name}] Epoch: {epoch} | Wall Time: {wall_time:.2f} | Epoch Time: {epoch_duration:.2f} | Loss: {np.mean(loss):.4f}"
            + (f" | Accuracy: {accuracy:.2f}%" if accuracy else "")
        )

    def configure_optimizers(self) -> None:
        self.scaler = None
        self.optimizers = []
        self.lr_schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
        params_lr = [(self.base_model.parameters(), self.config.lr)]
        if self.args.two_models:
            params_lr.append((self.overshoot_model.parameters(), self.config.lr * (self.args.overshoot_factor + 1)))
        elif self.config.precision == "16-mixed": # TODO: Scaler for two model is not working
            self.scaler = torch.GradScaler(device="cuda")

        for params, lr in params_lr:
            self.optimizers.append(
                create_optimizer(self.args.opt_name, params, self.args.overshoot_factor, lr, self.config)
            )
            if self.config.use_lr_scheduler:
                self.lr_schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizers[-1], T_0=self.steps + 1)
                )

    def save_stats(self) -> None:
        log_dir = self.log_writer.log_dir
        pd.DataFrame(self.train_stats).to_csv(os.path.join(log_dir, "training_stats.csv"), index=False)
        if self.val_stats:
            pd.DataFrame(self.val_stats).to_csv(os.path.join(log_dir, "validation_stats.csv"), index=False)
        if self.test_stats:
            pd.DataFrame(self.test_stats).to_csv(os.path.join(log_dir, "test_stats.csv"), index=False)

    def validation(self, epoch: int) -> float:
        self._set_model_mode(is_training=False)
        if self.args.compute_base_model_loss_validation:
            base_model, _ = self._get_base_model()
        else:
            base_model = self.base_model
        test_loaders: list[Tuple[DataLoader[UnifiedDatasetInterface], list[dict[str, float]], str]] = []
        if self.val_dataloader:
            test_loaders.append((self.val_dataloader, self.val_stats, "validation"))
        if self.test_dataloader:
            test_loaders.append((self.test_dataloader, self.test_stats, "test"))

        with torch.no_grad():
            for loader, stats, name in test_loaders:
                correct, total, loss = 0, 0, 0
                for batch in loader:
                    batch = self._move_batch_to_cuda(batch)
                    outputs = self._model_forward(base_model, batch)
                    _, predicted = outputs["logits"].max(1)
                    loss += outputs["loss"].item() * batch["labels"].size(0)
                    if loader.dataset.is_classification():  # type: ignore
                        total += batch["labels"].size(0)
                        correct += int(predicted.eq(batch["labels"]).sum().item())
                accuracy = 100 * correct / total if loader.dataset.is_classification() else None  # type: ignore
                self.log_stats(name, stats, epoch, loss / len(loader.dataset), accuracy)  # type: ignore
        self.save_stats()
        return loss / len(loader.dataset) # type: ignore

    def run(self) -> float:
        self.configure_optimizers()
        self.training_start_time = time.time()
        for epoch in range(self.config.epochs):
            self.epoch_start = time.time()

            # Training
            self._set_model_mode(is_training=True)
            for batch_id, batch in enumerate(self.train_dataloader):
                self._training_step(self._move_batch_to_cuda(batch), epoch, batch_id)
                self.current_step += self._is_update_batch(batch_id)
                if self.current_step >= self.steps:
                    print("Max steps reached. Finished training.")
                    return self.validation(epoch)

            loss = self.validation(epoch)
        return loss
