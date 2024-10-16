import torch
from dataclasses import dataclass
from typing import Optional, Sequence, get_type_hints, get_args
from collections import defaultdict


# Optimal LR: Roberta, sst, 3e-5 0.00003
#             CV tasks 0.001
#             MLP housing task 0.001 for batch = 16
#             MLP housing task 0.002 for batch = 64
# lr: float =  1e-5 # For LLM classification finetuning


def get_trainer_config(model_name: str, dataset_name: str, opt_name: str, override: Optional[Sequence[str]] = None):

    for x in ["sgd", "adam"]:
        if x in opt_name:
            opt_type = x
            break
    else:
        raise ValueError(f"Unsupported opt type: {opt_name}")
    
    return defaultdict(lambda: DefaultConfig, {
        ("mlp", "housing"): HousingConfig,
        ("mlp", "energy"): EnergyConfig,
        ("mlp", "mnist"): MlpMnistConfig,
        ("cnn", "mnist", "sgd"): CnnMnistSgdConfig,
        ("cnn", "mnist", "adam"): CnnMnistAdamConfig,
        ("gpt", "shakespear"): GptShakespearConfig,
        ("gpt", "gutenberg"): GptShakespearConfig,
    })[model_name, dataset_name, opt_type]().override(override)


@dataclass
class DefaultConfig:
    B: int = 64
    accumulate_grad_batches: int = 1
    lr: float = 0.001
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
    
    def override(self, override: Optional[Sequence[str]] = None) -> None:
        if override is None:
            return self
            
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
        return self



@dataclass
class HousingConfig(DefaultConfig):
    max_steps: int = 2000
    log_every_n_steps: int = 10
    mlp_hidden_size = [50]
        
@dataclass
class EnergyConfig(DefaultConfig):
    B: int = 32
    lr: float = 0.001
    epochs: int = 50
    log_every_n_steps: int = 10
    mlp_hidden_size = [8]
        

# Using this config to overfit the energy dataset
@dataclass
class EnergyConfig2(DefaultConfig):
    max_steps: int = 5000
    epochs: int = 999999
    log_every_n_steps: int = 50
    mlp_hidden_size = [50]
        
        
@dataclass
class MlpMnistConfig(DefaultConfig):
    epochs: int = 10
    mlp_hidden_size = [512, 256]
        
@dataclass
class CnnMnistSgdConfig(DefaultConfig):
    lr: float = 0.01
    epochs: int = 10
        
@dataclass
class CnnMnistAdamConfig(DefaultConfig):
    epochs: int = 10
        
@dataclass
class GptShakespearConfig(DefaultConfig):
    B: int = 16
    accumulate_grad_batches: int = 2
    epochs: int = 100
    lr: float = 3e-4
        
@dataclass
class GptGutenbergConfig(DefaultConfig):
    B: int = 16
    accumulate_grad_batches: int = 4
    epochs: int = 2
    lr: float = 3e-4