from typing import Optional

from transformers import (AutoConfig, AutoModelForPreTraining,
                          AutoModelForSequenceClassification, AutoTokenizer)

from cnn import CNN, ResNet
from mlp import MLP
from custom_datasets import (Cifar10Dataset, Cifar100Dataset, MMLUDataset,
                             MnistDataset, MNLIDataset, NextTokenDataloader,
                             QQPDataset, SST2Datatset, CaliforniaHousingDataset)
from gpt import GPT, GPTConfig, GPTTinyConfig
from trainer_configs import *


def init_model(model_name, dataset_name):
    model_map = {
        "gpt_hf": "openai-community/gpt2",
        "roberta_hf": "FacebookAI/roberta-base",
        "xlm_roberta_hf": "FacebookAI/xlm-roberta-base",
        "bloom_hf": "bigscience/bloom-560m",
        "mdeberta_hf": "microsoft/mdeberta-v3-base",
        "t5_hf": "google-t5/t5-base",
    }
    dataset_to_shape = {"mnist": ((28, 28, 3), 10), "cifar10": ((32, 32, 3), 10), "cifar100": ((32, 32, 3), 100), "housing": ((8,), 1)}

    if model_name == "gpt":
        tokenizer = AutoTokenizer.from_pretrained(model_map["gpt_hf"])  # use tokenizer from HF
        tokenizer.pad_token = tokenizer.eos_token
        return GPT(GPTConfig(vocab_size=50304)), tokenizer, 1024
    if model_name == "gpt_tiny":
        tokenizer = AutoTokenizer.from_pretrained(model_map["gpt_hf"])  # use tokenizer from HF
        tokenizer.pad_token = tokenizer.eos_token
        return GPT(GPTTinyConfig(vocab_size=50304)), tokenizer, 256
    elif model_name == "mlp":
        return MLP(dataset_to_shape[dataset_name][0], dataset_to_shape[dataset_name][1]), None, None
    elif model_name == "cnn":
        return CNN(dataset_to_shape[dataset_name][0], dataset_to_shape[dataset_name][1]), None, None
    elif model_name.startswith("resnet"):
        return ResNet(dataset_to_shape[dataset_name][1], type=model_name), None, None
    elif model_name in model_map:
        model_name = model_map[model_name]
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # config.hidden_dropout_prob = 0.0  # Default is 0.1
        # config.attention_probs_dropout_prob = 0.0  # Default is 0.1
        config.ignore_mismatched_sizes = True

        if dataset_name in ["shakespear", "gutenberg"]:
            model = AutoModelForPreTraining.from_config(config)  # from scratch
        else:
            config.num_labels = 3 if dataset_name == "mnli" else 2
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
        model.train()
        return model, tokenizer, 512
    else:
        raise ValueError(f"Model {model_name} not found")


def init_dataset(dataset_name, tokenizer: Optional = None, T: Optional = None):
    if dataset_name == "mnist":
        return MnistDataset()
    elif dataset_name == "cifar10":
        return Cifar10Dataset()
    elif dataset_name == "cifar100":
        return Cifar100Dataset()
    elif dataset_name == "shakespear":
        return NextTokenDataloader(tokenizer, T=T, source_file="tiny_shakespear_")
        # return NextTokenDataloader(tokenizer, T=T, source_file="tiny_shakespear.txt")
    elif dataset_name == "gutenberg":
        return NextTokenDataloader(tokenizer, T=T, source_file="gutenberg_books_")
    elif dataset_name == "sst":
        return SST2Datatset(tokenizer=tokenizer)
    elif dataset_name == "qqp":
        return QQPDataset(tokenizer=tokenizer)
    elif dataset_name == "mnli":
        return MNLIDataset(tokenizer=tokenizer)
    elif dataset_name == "mmlu":
        return MMLUDataset(tokenizer=tokenizer)
    elif dataset_name == "housing":
        return CaliforniaHousingDataset()
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

def get_gpu_stats(n_gpus: int = 0):
    gpu_info = ""
    for gpu_index in range(n_gpus):
        max_vram = torch.cuda.memory_reserved(gpu_index) / (1024 * 1024 * 1024)
        utilization = torch.cuda.utilization(gpu_index)
        gpu_info += f" | vram{gpu_index} {max_vram:.2f}GB | util{gpu_index} {utilization:.2f}%"
    return gpu_info
