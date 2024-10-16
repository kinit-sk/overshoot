import torch
from dataclasses import dataclass
from typing import Optional, Sequence, get_type_hints, get_args


# Optimal LR: Roberta, sst, 3e-5 0.00003
#             CV tasks 0.001
#             MLP housing task 0.001 for batch = 16
#             MLP housing task 0.002 for batch = 64


def get_trainer_config(model_name: str, dataset_name: str, override: Optional[Sequence[str]] = None):
    if model_name == "mlp" and dataset_name == "housing":
        return HousingConfig(override)

    return DefaultConfig(override)


@dataclass
class DefaultConfig:
    B: int = 16
    accumulate_grad_batches: int = 1
    # lr: float =  3e-4
    # lr: float =  1e-5 # For LLM classification finetuning
    lr: float = 0.001 # For CV
    # lr_overshoot: Optional[float] = None
    epochs: int = 15
    max_steps: Optional[int] = None
    decay_lr: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    sgd_momentum: float = 0.9
    weight_decay: float = 0.0
    log_every_n_steps: int = 50
    n_gpu: int = torch.cuda.device_count()
    use_16_bit_precision: bool = torch.cuda.device_count() > 0
    log_gpu: bool = False
    
    def __init__(self, override: Optional[Sequence[str]] = None) -> None:
        if override is None:
            return
            
        for key_value in override:
            key, value = key_value.split("=")
            if hasattr(self, key) == False:
                continue
            override_type = get_type_hints(DefaultConfig)[key]
            args = get_args(override_type)
            if len(args):
                override_type = args[0]
            if override_type is bool:
                if value == 'True':
                    setattr(self, key, True)
                elif value == 'False':
                    setattr(self, key, False)
            else:
                setattr(self, key, override_type(value))



@dataclass
class HousingConfig(DefaultConfig):
    B: int = 64
    lr: float = 0.001
    def __init__(self, override: Optional[Sequence[str]] = None) -> None:
        super().__init__(override=override)