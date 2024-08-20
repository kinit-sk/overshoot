import torch
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class CNNTrainerConfig:
    n_gpu: int = torch.cuda.device_count()  # Use all available gpus
    B: int = 256
    lr_base: float = 1e-4
    lr_overshoot: Optional[None] = None
    epochs: int = 6
    adam_betas: Tuple[float, float] = 0.9, 0.95
    weight_decay: float = 0.0
    accumulate_grad_batches: int = 2
    gradient_clip_val: float = 0.5
    
    
@dataclass
class RobertaTrainerConfig:
    n_gpu: int = torch.cuda.device_count()  # Use all available gpus
    B: int = 16
    lr_base: float = 6e-6
    lr_overshoot: Optional[None] = None
    epochs: int = 2
    adam_betas: Tuple[float, float] = 0.9, 0.95
    weight_decay: float = 0.0
    accumulate_grad_batches: int = 2
    gradient_clip_val: float = 0.5
    
    
@dataclass
class GPTTrainerConfig:
    n_gpu: int = torch.cuda.device_count()  # Use all available gpus
    B: int = 16
    lr_base: float = 3e-4
    lr_overshoot: Optional[None] = None
    epochs: int = 50
    adam_betas: Tuple[float, float] = 0.9, 0.95
    weight_decay: float = 0.1
    accumulate_grad_batches: int = 2
    gradient_clip_val: float = 0.5

@dataclass
class LLAMATrainerConfig:
    n_gpu: int = torch.cuda.device_count()  # Use all available gpus
    B: int = 1
    lr_base: float = 3e-4
    lr_overshoot: Optional[None] = None
    epochs: int = 2
    adam_betas: Tuple[float, float] = 0.9, 0.95
    weight_decay: float = 0.1
    accumulate_grad_batches: int = 2
    gradient_clip_val: float = 0.5