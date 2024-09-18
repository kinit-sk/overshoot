import torch
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainerConfig:
    n_gpu: int = torch.cuda.device_count()
    B: int = 64
    accumulate_grad_batches: int = 1
    # lr_base: float =  3e-4
    # lr_base: float =  1e-5 # For LLM classification finetuning
    lr_base: float = 0.001  # For CV problem use this 
    lr_overshoot: Optional[None] = None
    epochs: int = 15
    adam_betas: Tuple[float, float] = 0.9, 0.999
    sgd_momentum: float = 0.9
    weight_decay: float = 0.0

# @dataclass
# class CNNTrainerConfig:
#     n_gpu: int = torch.cuda.device_count()
#     B: int = 64
#     lr_base: float = 2e-3
#     lr_overshoot: Optional[None] = None
#     epochs: int = 160
#     adam_betas: Tuple[float, float] = 0.9, 0.95
#     weight_decay: float = 0.0
    
    
# @dataclass
# class RobertaTrainerConfig:
#     n_gpu: int = torch.cuda.device_count()
#     B: int = 16
#     lr_base: float = 3e-4
#     lr_overshoot: Optional[None] = None
#     epochs: int = 200
#     adam_betas: Tuple[float, float] = 0.9, 0.95
#     weight_decay: float = 0.0
    
    
# @dataclass
# class GPTTrainerConfig:
#     n_gpu: int = torch.cuda.device_count()
#     B: int = 16
#     lr_base: float = 3e-4
#     lr_overshoot: Optional[None] = None
#     epochs: int = 200
#     adam_betas: Tuple[float, float] = 0.9, 0.95
#     weight_decay: float = 0.1

# @dataclass
# class LLAMATrainerConfig:
#     n_gpu: int = torch.cuda.device_count()
#     B: int = 1
#     lr_base: float = 3e-4
#     lr_overshoot: Optional[None] = None
#     epochs: int = 2
#     adam_betas: Tuple[float, float] = 0.9, 0.95
#     weight_decay: float = 0.1