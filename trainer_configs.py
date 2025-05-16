import torch
from dataclasses import dataclass, field
from typing import Optional, Sequence, Literal, Callable, get_type_hints, get_args, Self
from collections import defaultdict

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
    epsilon: float = 1e-08
    sgd_momentum: float = 0.9
    weight_decay_sgd: float = 0.0 # pytorch default
    weight_decay_adam: float = 0.01 # pytorch default
    grad_clip: Optional[float] = 1.0
    overshoot_delay: int = 50
    optimizer_foreach: Optional[bool] = None
    target_cosine_similarity: float = 0.1
    log_every_n_steps: int = 50
    n_gpu: int = torch.cuda.device_count()
    precision: Literal["16-mixed", "default", "high"] = "16-mixed" if torch.cuda.device_count() > 0 else "default"
    use_peft: bool = True
    log_gpu: bool = False
    mlp_hidden_size: list = field(default_factory=lambda: [128, 64]) # type: ignore
    
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


def get_trainer_config(model_name: str, dataset_name: str, opt_name: str, override: Optional[Sequence[str]] = None, from_large_budget: bool = False) -> DefaultConfig:

    reduce: Callable[[str, str], str] = lambda x, substring: substring if substring in x else x
    model_name = reduce(model_name, "resnet")

    opt_name = reduce(opt_name, "nesterov")
    opt_name = reduce(opt_name, "sgd")
    opt_name = reduce(opt_name, "nadam")
    if opt_name != 'nadam':
        opt_name = reduce(opt_name, "adam")
    dataset_name = reduce(dataset_name, "cifar")

    # Supported only for
    #   1) mnist VAE
    #   2) fmnist VAE
    #   3) fmnist 2c2d
    #   4) cifar10 3c3d
    if from_large_budget:
        class_name = f"LargeBudget__CosineScheduler__{dataset_name}_{model_name}__{opt_name}__Config"
        if class_name in globals().keys():
            return globals()[class_name]().override(override) # type: ignore

    opt_name = "sgd" if opt_name == 'nesterov' else opt_name
    opt_name = "adam" if opt_name == 'nadam' else opt_name
        
    return defaultdict(lambda: DefaultConfig, {
        ("mlp", "boston", "sgd"): BostonConfig,
        ("mlp", "boston", "adam"): BostonConfig,
        ("mlp", "housing", "sgd"): HousingSgdConfig, ### TABLE 1: 1nd row config
        ("mlp", "housing", "adam"): HousingAdamConfig, ### TABLE 1: 1nd row config
        ("mlp", "energy", "sgd"): EnergyConfig,
        ("mlp", "energy", "adam"): EnergyConfig,
        ("mlp", "mnist", "sgd"): MlpMnistConfig,
        ("mlp", "mnist", "adam"): MlpMnistConfig, # TODO
        ("mlp", "cifar", "sgd"): MlpCifarSgdConfig,
        ("mlp", "cifar", "adam"): MlpCifarSgdConfig, # TODO
        ("2c2d", "mnist", "sgd"): _2c2dMnistSgdConfig,
        ("2c2d", "mnist", "adam"): _2c2dMnistAdamConfig,
        ("2c2d", "fmnist", "sgd"): _2c2dFashionSgdConfig, ### TABLE 1: 2nd row config
        ("2c2d", "fmnist", "adam"): _2c2dFashionAdamConfig, ### TABLE 1: 2nd row config
        ("3c3d", "cifar", "sgd"): _3c3dCifarSgdConfig, ### TABLE 1: 3nd row config
        ("3c3d", "cifar", "adam"): _3c3dCifarAdamConfig, ### TABLE 1: 3nd row config
        ("resnet", "mnist", "sgd"): ResnetMnistSgdConfig,
        ("resnet", "mnist", "adam"): ResnetMnistAdamConfig,
        ("resnet", "cifar", "sgd"): ResnetCifartSgdConfig,
        ("resnet", "cifar", "adam"): ResnetCifartAdamConfig,
        ("vae", "mnist", "sgd"): VaeMnistSgdConfig,
        ("vae", "mnist", "adam"): VaeMnistAdamConfig,
        ("vae", "fmnist", "sgd"): VaeFashionSgdConfig,
        ("vae", "fmnist", "adam"): VaeFashionAdamConfig,
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
@dataclass
class HousingSgdConfig(DefaultConfig):
    epochs: int = 200
    mlp_hidden_size = [200, 150]
    # Base on finetuning, validation loss as metric
    use_lr_scheduler: bool = True
    B: int = 32
    lr: float = 0.008
    sgd_momentum: float = 0.9
    
### TABLE 1: 1nd row config
@dataclass
class HousingAdamConfig(DefaultConfig):
    epochs: int = 200
    mlp_hidden_size = [200, 150]
    # Base on finetuning, validation loss as metric
    use_lr_scheduler: bool = True
    B: int = 64
    lr: float = 0.002
    adam_beta1: float = 0.95
        
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
    # Base on finetuning, validation accuracy as metric
    use_lr_scheduler: bool = True
    B: int = 32
    lr: float = 0.008
    sgd_momentum: float = 0.95
    
### TABLE 1: 2nd row config
# In paper `epochs=10`
@dataclass
class _2c2dFashionAdamConfig(DefaultConfig):
    epochs: int = 50
    # Base on finetuning, validation accuracy as metric
    use_lr_scheduler: bool = True
    B: int = 64
    lr: float = 0.001
    adam_beta1: float = 0.85
    
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
    weight_decay_sgd: float = 5e-4
    sgd_momentum: float = 0.99
    epochs: int = 250
    
### TABLE 1: 5nd row config
@dataclass
class ResnetCifartAdamConfig(DefaultConfig):
    B: int = 256
    weight_decay_adam: float = 5e-4
    epochs: int = 200
    
### TABLE 1: 4nd row config
@dataclass
class VaeMnistSgdConfig(DefaultConfig):
    epochs: int = 50
    # Base on finetuning, validation loss as metric
    use_lr_scheduler: bool = True
    B: int = 32
    lr: float = 0.002
    sgd_momentum: float = 0.85
    
### TABLE 1: 4nd row config
@dataclass
class VaeMnistAdamConfig(DefaultConfig):
    epochs: int = 50
    # Base on finetuning, validation loss as metric
    use_lr_scheduler: bool = True
    B: int = 32
    lr: float = 0.0005
    adam_beta1: float = 0.85
    
### TABLE 1: 5nd row config
@dataclass
class VaeFashionSgdConfig(DefaultConfig):
    epochs: int = 100
    # Base on finetuning, validation loss as metric
    use_lr_scheduler: bool = True
    B: int = 32
    lr: float = 0.002
    sgd_momentum: float = 0.85
    
### TABLE 1: 5nd row config
@dataclass
class VaeFashionAdamConfig(DefaultConfig):
    epochs: int = 100
    # Base on finetuning, validation loss as metric
    use_lr_scheduler: bool = True
    B: int = 64
    lr: float = 0.001
    adam_beta1: float = 0.85
    
    
################################################################################
########################### LLM classification tasks ###########################
################################################################################

@dataclass
class GptMnliSgdConfig(DefaultConfig):
    B: int = 64
    epochs: int = 20
    weight_decay_sgd: float = 5e-4
    
@dataclass
class GptMnliAdamConfig(DefaultConfig):
    B: int = 64
    epochs: int = 10
    weight_decay_adam: float = 5e-4


@dataclass
class GptSstAdamConfig(DefaultConfig):
    epochs: int = 20
    weight_decay_adam: float = 5e-4
    
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
    weight_decay_sgd: float = 5e-4
    weight_decay_adam: float = 5e-4
    lr: float = 3e-4
    
@dataclass
class BertQqpConfig(DefaultConfig):
    epochs: int = 20
    weight_decay_sgd: float = 5e-4
    weight_decay_adam: float = 5e-4
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
    weight_decay_adam: float = 5e-4
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



##############################################################
### Best configs from: https://github.com/SirRob1997/Crowded-Valley---Results/tree/master/results_main/large_budget/cosine

# FOR COSINE LR SCHEDULER!!!


@dataclass
class LargeBudget__CosineScheduler__mnist_vae__adam__Config(DefaultConfig):
    use_lr_scheduler: bool = True
    epochs: int = 50
    B: int = 64
    lr: float = 0.0006073170442405314
    weight_decay_adam: float = 0
    adam_beta1: float = 0.9242878251459865
    adam_beta2: float = 0.9488015600766657
    epsilon: float = 1e-08


@dataclass
class LargeBudget__CosineScheduler__mnist_vae__sgd__Config(DefaultConfig):
    use_lr_scheduler: bool = True
    epochs: int = 50
    B: int = 64
    lr: float = 0.0006522527301234561
    weight_decay_sgd: float = 0
    sgd_momentum: float = 0.8086103756739638


@dataclass
class LargeBudget__CosineScheduler__mnist_vae__nadam__Config(DefaultConfig):
    use_lr_scheduler: bool = True
    epochs: int = 50
    B: int = 64
    lr: float = 0.0012872772278229644
    weight_decay_adam: float = 0
    adam_beta1: float = 0.9269694894374103
    adam_beta2: float = 0.9666118685688961
    epsilon: float = 1e-07


@dataclass
class LargeBudget__CosineScheduler__mnist_vae__nesterov__Config(DefaultConfig):
    use_lr_scheduler: bool = True
    epochs: int = 50
    B: int = 64
    lr: float = 0.00314891164795686
    weight_decay_sgd: float = 0
    sgd_momentum: float = 0.3648778989359307


@dataclass
class LargeBudget__CosineScheduler__fmnist_vae__adam__Config(DefaultConfig):
    use_lr_scheduler: bool = True
    epochs: int = 100
    B: int = 64
    lr: float = 0.00044145368764944787
    weight_decay_adam: float = 0
    adam_beta1: float = 0.9515666733432182
    adam_beta2: float = 0.9573249396037167
    epsilon: float = 1e-08


@dataclass
class LargeBudget__CosineScheduler__fmnist_vae__sgd__Config(DefaultConfig):
    use_lr_scheduler: bool = True
    epochs: int = 100
    B: int = 64
    lr: float = 0.00045674144235604013
    weight_decay_sgd: float = 0
    sgd_momentum: float = 0.8933534298877723


@dataclass
class LargeBudget__CosineScheduler__fmnist_vae__nadam__Config(DefaultConfig):
    use_lr_scheduler: bool = True
    epochs: int = 100
    B: int = 64
    lr: float = 0.00044145368764944787
    weight_decay_adam: float = 0
    adam_beta1: float = 0.9515666733432182
    adam_beta2: float = 0.9573249396037167
    epsilon: float = 1e-07


@dataclass
class LargeBudget__CosineScheduler__fmnist_vae__nesterov__Config(DefaultConfig):
    use_lr_scheduler: bool = True
    epochs: int = 100
    B: int = 64
    lr: float = 0.0008295993771492153
    weight_decay_sgd: float = 0
    sgd_momentum: float = 0.9792937560684244


@dataclass
class LargeBudget__CosineScheduler__fmnist_2c2d__adam__Config(DefaultConfig):
    use_lr_scheduler: bool = True
    epochs: int = 100
    B: int = 128
    lr: float = 0.0005506590009122957
    weight_decay_adam: float = 0
    adam_beta1: float = 0.9228528401348194
    adam_beta2: float = 0.9313559489851898
    epsilon: float = 1e-08


@dataclass
class LargeBudget__CosineScheduler__fmnist_2c2d__sgd__Config(DefaultConfig):
    use_lr_scheduler: bool = True
    epochs: int = 100
    B: int = 128
    lr: float = 0.0007214950794832237
    weight_decay_sgd: float = 0
    sgd_momentum: float = 0.9965771793774238


@dataclass
class LargeBudget__CosineScheduler__fmnist_2c2d__nadam__Config(DefaultConfig):
    use_lr_scheduler: bool = True
    epochs: int = 100
    B: int = 128
    lr: float = 0.0005575453980775369
    weight_decay_adam: float = 0
    adam_beta1: float = 0.9274200018513628
    adam_beta2: float = 0.9018417694085219
    epsilon: float = 1e-07


@dataclass
class LargeBudget__CosineScheduler__fmnist_2c2d__nesterov__Config(DefaultConfig):
    use_lr_scheduler: bool = True
    epochs: int = 100
    B: int = 128
    lr: float = 0.000151673306880762
    weight_decay_sgd: float = 0
    sgd_momentum: float = 0.997998657937712


@dataclass
class LargeBudget__CosineScheduler__cifar_3c3d__adam__Config(DefaultConfig):
    use_lr_scheduler: bool = True
    epochs: int = 100
    B: int = 128
    lr: float = 0.000993502390906368
    weight_decay_adam: float = 0.002
    adam_beta1: float = 0.6642763568761334
    adam_beta2: float = 0.9462110561755975
    epsilon: float = 1e-08


@dataclass
class LargeBudget__CosineScheduler__cifar_3c3d__sgd__Config(DefaultConfig):
    use_lr_scheduler: bool = True
    epochs: int = 100
    B: int = 128
    lr: float = 0.0014742753159914664
    weight_decay_sgd: float = 0.002
    sgd_momentum: float = 0.9970795661528186


@dataclass
class LargeBudget__CosineScheduler__cifar_3c3d__nadam__Config(DefaultConfig):
    use_lr_scheduler: bool = True
    epochs: int = 100
    B: int = 128
    lr: float = 0.0010165510266418728
    weight_decay_adam: float = 0.002
    adam_beta1: float = 0.7054365920445971
    adam_beta2: float = 0.8553142950540596
    epsilon: float = 1e-07


@dataclass
class LargeBudget__CosineScheduler__cifar_3c3d__nesterov__Config(DefaultConfig):
    use_lr_scheduler: bool = True
    epochs: int = 100
    B: int = 128
    lr: float = 0.0014742753159914664
    weight_decay_sgd: float = 0.002
    sgd_momentum: float = 0.9970795661528186
