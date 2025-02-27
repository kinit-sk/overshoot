import torch
from dataclasses import dataclass
from typing import Optional, Sequence, Literal, get_type_hints, get_args, Self
from collections import defaultdict


# Optimal LR: Roberta, sst, 3e-5 0.00003
#             CV tasks 0.001
#             MLP housing task 0.001 for batch = 16
#             MLP housing task 0.002 for batch = 64
# lr: float =  1e-5 # For LLM classification finetuning


def get_trainer_config(model_name: str, dataset_name: str, opt_name: str, override: Optional[Sequence[str]] = None):

    reduce = lambda x, substring: substring if substring in x else x
    model_name = reduce(model_name, "resnet")
    opt_name = reduce(opt_name, "sgd")
    opt_name = reduce(opt_name, "adam")
    dataset_name = reduce(dataset_name, "cifar")
        
    return defaultdict(lambda: DefaultConfig, {
        ("mlp", "boston", "sgd"): BostonConfig,
        ("mlp", "boston", "adam"): BostonConfig,
        ("mlp", "housing", "sgd"): HousingConfig, ### TABLE 1: 1nd row config
        ("mlp", "housing", "adam"): HousingConfig, ### TABLE 1: 1nd row config
        ("mlp", "energy", "sgd"): EnergyConfig,
        ("mlp", "energy", "adam"): EnergyConfig,
        ("mlp", "mnist", "sgd"): MlpMnistConfig,
        ("mlp", "mnist", "adam"): MlpMnistConfig, # TODO
        ("mlp", "cifar", "sgd"): MlpCifarSgdConfig,
        ("mlp", "cifar", "adam"): MlpCifarSgdConfig, # TODO
        ("2c2d", "mnist", "sgd"): _2c2dMnistSgdConfig,
        ("2c2d", "mnist", "adam"): _2c2dMnistAdamConfig,
        ("2c2d", "f-mnist", "sgd"): _2c2dFashionSgdConfig, ### TABLE 1: 2nd row config
        ("2c2d", "f-mnist", "adam"): _2c2dFashionAdamConfig, ### TABLE 1: 2nd row config
        ("3c3d", "cifar", "sgd"): _3c3dCifarSgdConfig, ### TABLE 1: 3nd row config
        ("3c3d", "cifar", "adam"): _3c3dCifarAdamConfig, ### TABLE 1: 3nd row config
        ("resnet", "mnist", "sgd"): ResnetMnistSgdConfig,
        ("resnet", "mnist", "adam"): ResnetMnistAdamConfig,
        ("resnet", "cifar", "sgd"): ResnetCifartSgdConfig,
        ("resnet", "cifar", "adam"): ResnetCifartAdamConfig,
        ("vae", "mnist", "sgd"): VaeMnistConfig,
        ("vae", "mnist", "adam"): VaeMnistConfig,
        ("vae", "f-mnist", "sgd"): VaeFashionConfig,
        ("vae", "f-mnist", "adam"): VaeFashionConfig,
        ("gpt_hf", "mnli", "sgd"): GptMnliSgdConfig,
        ("gpt_hf", "mnli", "adam"): GptMnliAdamConfig,
        ("gpt_hf", "sst", "adam"): GptSstAdamConfig,
        ("roberta_hf", "sst", "adam"): RobertaSstAdamConfig,
        ("minilm", "sst", "adam"): MinilmSstAdamConfig,
        ("bloom_hf", "sst", "adam"): BloomSstAdamConfig,
        ("gpt_hf", "qqp", "sgd"): GptQqpConfig,
        ("gpt_hf", "qqp", "adam"): GptQqpConfig,
        ("bert_hf", "qqp", "sgd"): BertQqpConfig,
        ("bert_hf", "qqp", "adam"): BertQqpConfig,
        ("bert_hf", "imdb", "adam"): BertImdbConfig,
        ("roberta_hf", "qqp", "adam"): RobertaQqpAdamConfig,
        ("gpt", "shakespear", "sgd"): GptShakespearConfig, # TODO
        ("gpt", "shakespear", "adam"): GptShakespearConfig,
        ("gpt", "gutenberg", "sgd"): GptGutenbergConfig, # TODO
        ("gpt", "gutenberg", "adam"): GptGutenbergConfig,
    })[model_name, dataset_name, opt_name]().override(override)


@dataclass
class DefaultConfig:
    B: int = 64
    accumulate_grad_batches: int = 1
    lr: float = 0.001
    epochs: int = 50
    max_steps: Optional[int] = None
    use_lr_scheduler: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    sgd_momentum: float = 0.9
    weight_decay: float = 0.0
    overshoot_delay: int = 50
    optimizer_foreach: Optional[bool] = None
    target_cosine_similarity: float = 0.1
    log_every_n_steps: int = 50
    n_gpu: int = torch.cuda.device_count()
    precision: Literal["16-mixed", "default", "high"] = "16-mixed" if torch.cuda.device_count() > 0 else "default"
    use_peft: bool = True
    log_gpu: bool = False
    
    def override(self, override: Optional[Sequence[str]] = None) -> Self:
        if override is None:
            return self
            
        for key_value in override:
            key, value = key_value.split("=")
            if hasattr(self, key) == False:
                continue
            override_type = get_type_hints(DefaultConfig)[key]
            args = get_args(override_type)
            
            if override_type.__name__ == 'Literal':
                if not value in args:
                    raise ValueError(f"{key} must be one of {args}")
                else:
                    setattr(self, key, value)
                    continue
                    
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

### TABLE 1: 1nd row config
# When using SGD we do not manage to overfit, but that is fine...
@dataclass
class BostonConfig(DefaultConfig):
    B: int = 32
    epochs: int = 100
    mlp_hidden_size = [128, 64]
    
### TABLE 1: 1nd row config
# When using SGD we do not manage to overfit, but that is fine...
@dataclass
class HousingConfig(DefaultConfig):
    epochs: int = 200
    mlp_hidden_size = [200, 150]
        
@dataclass
class EnergyConfig(DefaultConfig):
    B: int = 32
    epochs: int = 250
    mlp_hidden_size = [100, 50]
        

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
    epochs: int = 50
    mlp_hidden_size = [512, 256]
    
@dataclass
class MlpCifarSgdConfig(DefaultConfig):
    lr: float = 5e-3
    epochs: int = 5
    mlp_hidden_size = [512, 256]
    
        
@dataclass
class _2c2dMnistSgdConfig(DefaultConfig):
    lr: float = 0.01
    epochs: int = 50
        
@dataclass
class _2c2dMnistAdamConfig(DefaultConfig):
    epochs: int = 50
    
### TABLE 1: 2nd row config
@dataclass
class _2c2dFashionSgdConfig(DefaultConfig):
    epochs: int = 50
    
### TABLE 1: 2nd row config
@dataclass
class _2c2dFashionAdamConfig(DefaultConfig):
    epochs: int = 10
    
### TABLE 1: 3nd row config
@dataclass
class _3c3dCifarSgdConfig(DefaultConfig):
    B: int = 128
    lr: float = 0.01
    epochs: int = 100
    
### TABLE 1: 3nd row config
@dataclass
class _3c3dCifarAdamConfig(DefaultConfig):
    B: int = 128
    epochs: int = 100
    
    
@dataclass
class ResnetMnistSgdConfig(DefaultConfig):
    lr: float = 2e-3
    epochs: int = 2
    
@dataclass
class ResnetMnistAdamConfig(DefaultConfig):
    lr: float = 5e-4
    epochs: int = 2
    
### TABLE 1: 5nd row config
@dataclass
class ResnetCifartSgdConfig(DefaultConfig):
    B: int = 256
    lr: float = 0.01
    weight_decay: float = 5e-4
    sgd_momentum: float = 0.99
    epochs: int = 250
    
### TABLE 1: 5nd row config
@dataclass
class ResnetCifartAdamConfig(DefaultConfig):
    B: int = 256
    weight_decay: float = 5e-4
    epochs: int = 200
    
### TABLE 1: 4nd row config
@dataclass
class VaeMnistConfig(DefaultConfig):
    epochs: int = 50
    
### TABLE 1: 5nd row config
@dataclass
class VaeFashionConfig(DefaultConfig):
    epochs: int = 100
    
    
################################################################################
########################### LLM classification tasks ###########################
################################################################################

@dataclass
class GptMnliSgdConfig(DefaultConfig):
    B: int = 64
    epochs: int = 20
    weight_decay: float = 5e-4
    
@dataclass
class GptMnliAdamConfig(DefaultConfig):
    B: int = 64
    epochs: int = 10
    weight_decay: float = 5e-4


@dataclass
class GptSstAdamConfig(DefaultConfig):
    epochs: int = 20
    weight_decay: float = 5e-4
    
@dataclass
class RobertaSstAdamConfig(DefaultConfig):
    lr: float = 2e-5
    epochs: int = 10
    
@dataclass
class BloomSstAdamConfig(DefaultConfig):
    lr: float = 2e-5
    epochs: int = 2
    
@dataclass
class MinilmSstAdamConfig(DefaultConfig):
    lr: float = 2e-5
    epochs: int = 6
    
    
@dataclass
class GptQqpConfig(DefaultConfig):
    epochs: int = 10
    weight_decay: float = 5e-4
    lr: float = 3e-4
    
@dataclass
class BertQqpConfig(DefaultConfig):
    epochs: int = 20
    weight_decay: float = 5e-4
    lr: float = 3e-4
    
@dataclass
class BertImdbConfig(DefaultConfig):
    epochs: int = 10
    # weight_decay: float = 5e-4
    lr: float = 5e-5
    use_peft: bool = False
    
@dataclass
class RobertaQqpAdamConfig(DefaultConfig):
    epochs: int = 10
    weight_decay: float = 5e-4
    lr: float = 3e-4


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
    max_steps: int  = 6000
