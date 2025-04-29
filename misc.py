from typing import Any, Callable, Iterator, Optional

import torch
import numpy as np
from torch.optim import SGD, Adam, AdamW, NAdam, RMSprop # type: ignore
from torch.optim.optimizer import Optimizer
from peft import LoraConfig, TaskType, get_peft_model # type: ignore
from transformers import (AutoConfig, AutoModelForPreTraining,
                          AutoModelForSequenceClassification, AutoTokenizer)

from custom_datasets import (NextTokenDataloader, UnifiedDatasetInterface, create_boston_datatset,
                             create_cifar, create_energy_datatset, create_imbd,
                             create_fasion_mnist, create_housing_datatset,
                             create_mnist, create_mnli, create_qqp, create_sst, DatasetType)
from trainer_configs import DefaultConfig
from models._2c2d import _2c2d
from models._3c3d import _3c3d
from models.gpt import GPT, GPTConfig, GPTTinyConfig # type: ignore
from models.mlp import MLP
from models.resnet import ResNet
from models.vae import VAE
from trainer_configs import *

from overshoot.sgd_overshoot import SGDO
from overshoot.adamw_overshoot_delayed import AdamO as OvershootAdamW_delayed

from optimizers_old.backups2.sgdo_adaptive import SGDO as SGDO_adaptive
from optimizers_old.backups2.adamw_overshoot_replication import AdamW as OvershootAdamW_replication
from optimizers_old.backups2.adamw_overshoot_full_approximation import AdamW as OvershootAdamW_full_approximation
from optimizers_old.backups2.adamw_overshoot_denom_approximation import AdamW as OvershootAdamW_denom_approximation
from optimizers_old.backups2.adamw_overshoot_adaptive import AdamW as OvershootAdamW_adaptive

supported_datasets = [
    "mnist",
    "fmnist",
    "cifar10",
    "cifar100",
    "housing",
    "sst",
    "qqp",
    "mnli",
    "shakespear",
]

supported_models = [
    "mlp",
    "2c2d",
    "3c3d",
    "resnet18",
    "resnet50",
    "vae",
    "gpt",
    "roberta_hf",
    "bloom_hf",
    "minilm",
]

optimizers_map: dict[str, Callable[..., Optimizer]] = {
    "sgd": SGD,
    "sgd_momentum": SGD,
    "sgd_nesterov": SGD,
    "sgd_overshoot": SGDO,
    "sgd_adaptive": SGDO_adaptive,
    "adam": Adam,
    "adamW": AdamW,
    "adam_zero": Adam,
    "adamW_zero": AdamW,
    "nadam": NAdam,
    "adamW_overshoot_replication": OvershootAdamW_replication,
    "adamW_overshoot_full_approximation": OvershootAdamW_full_approximation,
    "adamW_overshoot_denom_approximation": OvershootAdamW_denom_approximation,
    "adamW_overshoot_delayed": OvershootAdamW_delayed,
    "adamW_overshoot_adaptive": OvershootAdamW_adaptive,
    "rmsprop": RMSprop,
}


def init_dataset(dataset_name: str, model_name: Optional[str], seed: Optional[int] = None) -> DatasetType:
    if dataset_name == "mnist":
        return create_mnist(model_name == "vae")
    elif dataset_name == "cifar10":
        return create_cifar(10)
    elif dataset_name == "cifar100":
        return create_cifar(100)
    elif dataset_name == "boston":
        return create_boston_datatset(seed=seed if seed else 42)
    elif dataset_name == "housing":
        return create_housing_datatset()
        # return create_housing_datatset(seed=seed if seed else 42)
    elif dataset_name == "energy":
        return create_energy_datatset()
    elif dataset_name == "fmnist":
        return create_fasion_mnist(model_name == "vae")

    assert model_name
    model_map = {
        "gpt_hf": "openai-community/gpt2",
        "gpt": "openai-community/gpt2",
        "gpt_tiny": "openai-community/gpt2",
        "bert_hf": "google-bert/bert-base-uncased",
        "roberta_hf": "FacebookAI/roberta-base",
        "xlm_roberta_hf": "FacebookAI/xlm-roberta-base",
        "bloom_hf": "bigscience/bloom-560m",
        "mdeberta_hf": "microsoft/mdeberta-v3-base",
        "t5_hf": "google-t5/t5-base",
        "minilm": "microsoft/MiniLM-L12-H384-uncased",
    }
    context_map = {
        "gpt_hf": 1024,
        "gpt": 1024,
        "gpt_tiny": 256,
        "bert_hf": 512,
        "roberta_hf": 512,
        "xlm_roberta_hf": 512,
        "bloom_hf": 512,
        "mdeberta_hf": 512,
        "t5_hf": 512,
        "minilm": 512,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_map[model_name])
    if (tokenizer.pad_token is None) and (tokenizer.eos_token is not None):
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_name == "shakespear":
        # return NextTokenDataloader(tokenizer, T=context_map[model_name], source_file="tiny_shakespear_")
        return NextTokenDataloader(tokenizer, T=context_map[model_name], source_file="tiny_shakespear.txt"), None, None
    elif dataset_name == "gutenberg":
        return NextTokenDataloader(tokenizer, T=context_map[model_name], source_file="gutenberg_books_"), None, None
    elif dataset_name == "sst":
        return create_sst(tokenizer)
    elif dataset_name == "qqp":
        return create_qqp(tokenizer)
    elif dataset_name == "imdb":
        return create_imbd(tokenizer)
    elif dataset_name == "mnli":
        return create_mnli(tokenizer=tokenizer)
    # TODO:
    # elif dataset_name == "mmlu":
    #     return MMLUDataset(tokenizer=tokenizer)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")


def init_model(model_name: str, datatset: UnifiedDatasetInterface, trainer_config: DefaultConfig) -> torch.nn.Module:
    model_map = {
        "gpt_hf": "openai-community/gpt2",
        "bert_hf": "google-bert/bert-base-uncased",
        "roberta_hf": "FacebookAI/roberta-base",
        "xlm_roberta_hf": "FacebookAI/xlm-roberta-base",
        "bloom_hf": "bigscience/bloom-560m",
        "mdeberta_hf": "microsoft/mdeberta-v3-base",
        "t5_hf": "google-t5/t5-base",
        "minilm": "microsoft/MiniLM-L12-H384-uncased",
    }
    n_outputs = datatset.n_outputs()

    if isinstance(datatset[0], dict):
        if model_name == "gpt":
            return GPT(GPTConfig(vocab_size=50304)) # type: ignore
        if model_name == "gpt_tiny":
            return GPT(GPTTinyConfig(vocab_size=50304)) # type: ignore
        elif model_name == "mlp":
            # If `mlp_hidden_size` is not set assign default one
            if not hasattr(trainer_config, "mlp_hidden_size"):
                trainer_config.mlp_hidden_size = [512, 256]
            inpt_shape = datatset[0]["x"].shape
            return MLP(inpt_shape, n_outputs, datatset.is_classification(), hidden_layers=trainer_config.mlp_hidden_size)
        elif model_name == "2c2d":
            inpt_shape = datatset[0]["x"].shape
            return _2c2d(inpt_shape, n_outputs)
        elif model_name == "3c3d":
            inpt_shape = datatset[0]["x"].shape
            return _3c3d(inpt_shape, n_outputs)
        elif model_name.startswith("resnet"):
            return ResNet(n_outputs, type=model_name)
        elif model_name == "vae":
            return VAE()
        elif model_name in model_map:
            model_name = model_map[model_name]
            config = AutoConfig.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Keep dropout
        # if "gpt" in model_name:
        #     config.resid_pdrop = 0
        #     config.embd_pdrop = 0
        #     config.attn_pdrop = 0
        # else:
        #     config.hidden_dropout_prob = 0.0  # Default is 0.1
        #     config.attention_probs_dropout_prob = 0.0  # Default is 0.1
        config.ignore_mismatched_sizes = True

        if isinstance(datatset, NextTokenDataloader):
            model = AutoModelForPreTraining.from_config(config)  # from scratch
        else:
            # config.num_labels = 3 if isinstance(datatset, MNLIDataset) else 2
            config.num_labels = n_outputs
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

        if (tokenizer.pad_token is None) and (tokenizer.eos_token is not None):
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]

        if trainer_config.use_peft:
            peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=6, lora_dropout=0.1)
            model = get_peft_model(model, peft_config)
            print("Using peft:")
            model.print_trainable_parameters()
        return model # type: ignore
    else:
        raise ValueError(f"Model {model_name} not found")


def get_gpu_stats(n_gpus: int = 0) -> str:
    gpu_info = ""
    for gpu_index in range(n_gpus):
        max_vram = torch.cuda.memory_reserved(gpu_index) / (1024 * 1024 * 1024)
        utilization = torch.cuda.utilization(gpu_index)
        gpu_info += f" | vram{gpu_index} {max_vram:.2f}GB | util{gpu_index} {utilization:.2f}%"
    return gpu_info


def compute_model_distance(ref_model: torch.Tensor, gradient_models: list[torch.Tensor], decay_factor: float) -> float:
    assert 0 < decay_factor < 1
    return float(sum(
        [np.linalg.norm(ref_model - g_m) * decay_factor**i for i, g_m in enumerate(reversed(gradient_models))]
    ))


def get_model_size(model: torch.nn.Module) -> float:
    param_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4  # Assuming float32
    buffer_size = sum(p.numel() for p in model.buffers()) * 4
    size_all_mb = (param_size + buffer_size) / 1024 / 1024
    return round(size_all_mb, 2)

def create_optimizer(opt_name: str, param_groups: Iterator[torch.nn.parameter.Parameter], overshoot_factor: float, lr: float, config: DefaultConfig) -> Optimizer:
    if opt_name == "nadam":
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.epsilon,
            momentum_decay=1000000000000000000000000, # Turn of momentum decay
            weight_decay=config.weight_decay,
            decoupled_weight_decay=True,
            foreach=config.optimizer_foreach,
        )
    elif opt_name == "adamW_overshoot_delayed":
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
            overshoot=overshoot_factor,
            overshoot_delay=config.overshoot_delay,
            foreach=config.optimizer_foreach,
        )
    elif opt_name == "adamW_overshoot_adaptive":
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
            cosine_target=config.target_cosine_similarity,
            foreach=config.optimizer_foreach,
        )
    elif opt_name.startswith("adamW_overshoot"):
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
            overshoot=overshoot_factor,
            foreach=config.optimizer_foreach,
        )
    elif "adam" in opt_name:
        adam_beta1 = 0 if "zero" in opt_name else config.adam_beta1
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            betas=(adam_beta1, config.adam_beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
            foreach=config.optimizer_foreach,
        )
    elif "sgd_adaptive" in opt_name:
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay,
            cosine_target=config.target_cosine_similarity,
            foreach=config.optimizer_foreach,
        )
    elif "sgd_overshoot" in opt_name:
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay,
            overshoot=overshoot_factor,
            foreach=config.optimizer_foreach,
        )
    elif "sgd" in opt_name:
        opt = optimizers_map[opt_name](
            param_groups,
            lr=lr,
            momentum=0 if opt_name == "sgd" else config.sgd_momentum,
            weight_decay=config.weight_decay,
            nesterov="nesterov" in opt_name,
            foreach=config.optimizer_foreach,
        )
    else:
        raise Exception(f"Optimizer {opt_name} not recognized.")
    return opt
