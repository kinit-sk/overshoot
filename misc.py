from typing import Optional, List
import numpy as np

from transformers import (AutoConfig, AutoModelForPreTraining,
                          AutoModelForSequenceClassification, AutoTokenizer)

from cnn import CNN, ResNet
from mlp import MLP
from custom_datasets import (MMLUDataset, MNLIDataset, NextTokenDataloader,
                             QQPDataset, SST2Datatset, create_mnist, create_cifar, create_housing_datatset, create_energy_datatset)
from gpt import GPT, GPTConfig, GPTTinyConfig
from trainer_configs import *




def init_dataset(dataset_name, model_name: Optional[str]):
    if dataset_name == "mnist":
        return create_mnist()
    elif dataset_name == "cifar10":
        return create_cifar(10)
    elif dataset_name == "cifar100":
        return create_cifar(100)
    elif dataset_name == "housing":
        return create_housing_datatset()
    elif dataset_name == "energy":
        return create_energy_datatset()
        
    assert model_name
    model_map = {
        "gpt_hf": "openai-community/gpt2",
        "gpt": "openai-community/gpt2",
        "gpt_tiny": "openai-community/gpt2",
        "roberta_hf": "FacebookAI/roberta-base",
        "xlm_roberta_hf": "FacebookAI/xlm-roberta-base",
        "bloom_hf": "bigscience/bloom-560m",
        "mdeberta_hf": "microsoft/mdeberta-v3-base",
        "t5_hf": "google-t5/t5-base",
    }
    context_map = {
        "gpt_hf": 1024,
        "gpt": 1024,
        "gpt_tiny": 256,
        "roberta_hf": 512,
        "xlm_roberta_hf": 512,
        "bloom_hf": 512,
        "mdeberta_hf": 512,
        "t5_hf": 512,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_map[model_name])
    tokenizer.pad_token = tokenizer.eos_token
        
    if dataset_name == "shakespear":
        return NextTokenDataloader(tokenizer, T=context_map[model_name], source_file="tiny_shakespear_")
        # return NextTokenDataloader(tokenizer, T=T, source_file="tiny_shakespear.txt")
    elif dataset_name == "gutenberg":
        return NextTokenDataloader(tokenizer, T=context_map[model_name], source_file="gutenberg_books_")
    elif dataset_name == "sst":
        return SST2Datatset(tokenizer=tokenizer)
    elif dataset_name == "qqp":
        return QQPDataset(tokenizer=tokenizer)
    elif dataset_name == "mnli":
        return MNLIDataset(tokenizer=tokenizer)
    elif dataset_name == "mmlu":
        return MMLUDataset(tokenizer=tokenizer)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
        
        
def init_model(model_name, datatset, trainer_config):
    model_map = {
        "gpt_hf": "openai-community/gpt2",
        "roberta_hf": "FacebookAI/roberta-base",
        "xlm_roberta_hf": "FacebookAI/xlm-roberta-base",
        "bloom_hf": "bigscience/bloom-560m",
        "mdeberta_hf": "microsoft/mdeberta-v3-base",
        "t5_hf": "google-t5/t5-base",
    }
    n_outputs = datatset.n_outputs()

    if model_name == "gpt":
        return GPT(GPTConfig(vocab_size=50304))
    if model_name == "gpt_tiny":
        return GPT(GPTTinyConfig(vocab_size=50304))
    elif model_name == "mlp":
        inpt_shape = datatset[0]["x"].shape
        return MLP(inpt_shape, n_outputs, datatset.is_classification(), hidden_layers=trainer_config.mlp_hidden_size)
    elif model_name == "cnn":
        inpt_shape = datatset[0]["x"].shape
        return CNN(inpt_shape, n_outputs)
    elif model_name.startswith("resnet"):
        return ResNet(n_outputs, type=model_name)
    elif model_name in model_map:
        model_name = model_map[model_name]
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if "gpt" in model_name:
            config.resid_pdrop = 0
            config.embd_pdrop = 0
            config.attn_pdrop = 0
        else:
            config.hidden_dropout_prob = 0.0  # Default is 0.1
            config.attention_probs_dropout_prob = 0.0  # Default is 0.1
        config.ignore_mismatched_sizes = True

        if isinstance(datatset, NextTokenDataloader):
            model = AutoModelForPreTraining.from_config(config)  # from scratch
        else:
            config.num_labels = 3 if isinstance(datatset, MNLIDataset) else 2
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
        model.train()
        return model
    else:
        raise ValueError(f"Model {model_name} not found")

def get_gpu_stats(n_gpus: int = 0):
    gpu_info = ""
    for gpu_index in range(n_gpus):
        max_vram = torch.cuda.memory_reserved(gpu_index) / (1024 * 1024 * 1024)
        utilization = torch.cuda.utilization(gpu_index)
        gpu_info += f" | vram{gpu_index} {max_vram:.2f}GB | util{gpu_index} {utilization:.2f}%"
    return gpu_info


def compute_model_distance(ref_model: torch.Tensor, gradient_models: List[torch.Tensor], m: float):
    return sum([np.linalg.norm(ref_model - g_m) * m**i for i, g_m in enumerate(reversed(gradient_models))])
