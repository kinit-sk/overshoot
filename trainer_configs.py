import torch
from dataclasses import dataclass
from typing import Optional, Sequence, get_type_hints, get_args
from collections import defaultdict


# Optimal LR: Roberta, sst, 3e-5 0.00003
#             CV tasks 0.001
#             MLP housing task 0.001 for batch = 16
#             MLP housing task 0.002 for batch = 64
# lr: float =  1e-5 # For LLM classification finetuning


def get_trainer_config(model_name: str, dataset_name: str, opt_name: str, use_high_precision: bool, override: Optional[Sequence[str]] = None):

    reduce = lambda x, substring: substring if substring in x else x
    model_name = reduce(model_name, "resnet")
    opt_name = reduce(opt_name, "sgd")
    opt_name = reduce(opt_name, "adam")
    dataset_name = reduce(dataset_name, "cifar")
        
    cfg = defaultdict(lambda: DefaultConfig, {
        ("mlp", "housing", "sgd"): HousingConfig,
        ("mlp", "housing", "adam"): HousingConfig,
        ("mlp", "energy", "sgd"): EnergyConfig,
        ("mlp", "energy", "adam"): EnergyConfig,
        ("mlp", "mnist", "sgd"): MlpMnistConfig,
        ("mlp", "mnist", "adam"): MlpMnistConfig, # TODO
        ("mlp", "cifar", "sgd"): MlpCifarSgdConfig,
        ("mlp", "cifar", "adam"): MlpCifarSgdConfig, # TODO
        ("cnn", "mnist", "sgd"): CnnMnistSgdConfig,
        ("cnn", "mnist", "adam"): CnnMnistAdamConfig,
        ("cnn", "cifar", "sgd"): CnnCifarSgdConfig,
        ("cnn", "cifar", "adam"): CnnCifarAdamConfig,
        ("resnet", "mnist", "sgd"): ResnetMnistSgdConfig,
        ("resnet", "mnist", "adam"): ResnetMnistAdamConfig,
        ("resnet", "cifar", "sgd"): ResnetCifartSgdConfig,
        ("resnet", "cifar", "adam"): ResnetCifartAdamConfig,
        ("gpt_hf", "sst", "adam"): GptSstAdamConfig,
        ("roberta_hf", "sst", "adam"): RobertaSstAdamConfig,
        ("bloom_hf", "sst", "adam"): BloomSstAdamConfig,
        ("gpt_hf", "qqp", "adam"): GptQqpAdamConfig,
        ("roberta_hf", "qqp", "adam"): RobertaQqpAdamConfig,
        ("gpt", "shakespear"): GptShakespearConfig,
        ("gpt", "gutenberg"): GptShakespearConfig,
    })[model_name, dataset_name, opt_name]().override(override)

    if use_high_precision:
        cfg.use_16_bit_precision = False
    return cfg


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
    overshoot_delay: int = 50
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


################################################################################
############################# MLP regression tasks #############################
################################################################################

@dataclass
class HousingConfig(DefaultConfig):
    epochs: int = 200
    log_every_n_steps: int = 50
    mlp_hidden_size = [200, 150]
        
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
        
        
################################################################################
############################# Computer vision tasks ############################
################################################################################

@dataclass
class MlpMnistConfig(DefaultConfig):
    epochs: int = 10
    mlp_hidden_size = [512, 256]
    
@dataclass
class MlpCifarSgdConfig(DefaultConfig):
    lr: float = 5e-3
    epochs: int = 5
    mlp_hidden_size = [512, 256]
    
        
@dataclass
class CnnMnistSgdConfig(DefaultConfig):
    lr: float = 0.01
    epochs: int = 10
        
@dataclass
class CnnMnistAdamConfig(DefaultConfig):
    epochs: int = 10
    
@dataclass
class CnnCifarSgdConfig(DefaultConfig):
    epochs: int = 15
    
@dataclass
class CnnCifarAdamConfig(DefaultConfig):
    lr: float = 1e-3
    epochs: int = 15
    
    
@dataclass
class ResnetMnistSgdConfig(DefaultConfig):
    lr: float = 2e-3
    epochs: int = 2
    
@dataclass
class ResnetMnistAdamConfig(DefaultConfig):
    lr: float = 5e-4
    epochs: int = 2
    
@dataclass
class ResnetCifartSgdConfig(DefaultConfig):
    lr: float = 2e-4 # 4e-3 for cira100
    epochs: int = 20 # 20 for cira100
    
@dataclass
class ResnetCifartAdamConfig(DefaultConfig):
    lr: float = 2e-4
    epochs: int = 20
    
    
################################################################################
########################### LLM classification tasks ###########################
################################################################################


@dataclass
class GptSstAdamConfig(DefaultConfig):
    lr: float = 5e-5
    epochs: int = 3
    log_every_n_steps: int = 5
    
@dataclass
class RobertaSstAdamConfig(DefaultConfig):
    lr: float = 2e-5
    epochs: int = 2
    log_every_n_steps: int = 5
    
@dataclass
class BloomSstAdamConfig(DefaultConfig):
    lr: float = 2e-5
    epochs: int = 2
    log_every_n_steps: int = 5
    
    
@dataclass
class GptQqpAdamConfig(DefaultConfig):
    lr: float = 5e-5
    epochs: int = 3
    log_every_n_steps: int = 5
    
@dataclass
class RobertaQqpAdamConfig(DefaultConfig):
    lr: float = 2e-5
    epochs: int = 2
    log_every_n_steps: int = 5


################################################################################
########################## Next token prediction tasks #########################
################################################################################


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