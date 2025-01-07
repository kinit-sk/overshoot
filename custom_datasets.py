import os
from functools import partial
from typing import Callable, Optional, Self

import tiktoken
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import random_split, Dataset
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datasets import load_dataset

from transformers import AutoTokenizer

class NextTokenDataloader:
    
    def __init__(self, tokenizer, T: int, source_file: str = 'tiny_shakespear.txt', cache_dir='.next-token-dataloader'):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
        self.T = T
        self._tokenizer = tokenizer

        file_path = os.path.join(cache_dir, source_file)
        # Download source file if needed
        if not os.path.exists(file_path) and all([not f.startswith(source_file) for f in os.listdir(cache_dir)]):
            download_links = {
                "tiny_shakespear.txt": 'https://drive.google.com/uc?id=1zPB3Y_9mUfKTrywpOX5Jz9j7TwcMkbnC',
                "gutenberg_books.txt": 'https://drive.google.com/uc?id=10N-sj1Nu2dj6XjyBxi43tfqamxb9Rn6t'
            }
            if source_file not in download_links:
                raise Exception(f'Unsupported source file: {source_file}')
            import gdown
            gdown.download(download_links[source_file], file_path, quiet=False)

        self._current_shred = 0
        self._shred_offset = 0
        self._length = 0
        if source_file.endswith("_"):
            self._sherds = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.startswith(source_file) and 'split' in f]
            if os.path.exists(os.path.join(cache_dir, f"{source_file}meta.yaml")):
                import yaml
                # meta_data = yaml.safe_load(os.path.join(cache_dir, f"{source_file}meta.yaml"))
                with open(os.path.join(cache_dir, f"{source_file}meta.yaml"), 'r') as f:
                    meta_data = yaml.safe_load(f)
                self._length = sum([int(x) for x in meta_data['sizes'].split(' ')])
                print(f"Loaded {self._length} tokens")
                return
        else:
            self._sherds = [file_path]
            
        modulo = 0
        for i, s in enumerate(self._sherds):
            tokens = self.__tokenize_file(s)
            self._length += (modulo + tokens.shape[0]) // self.T
            modulo = (modulo + tokens.shape[0]) % self.T
            print(f"Loading shred: {i}/{len(self._sherds)}, size: {tokens.shape[0] // self.T}", flush=True)
            if i == 0:
                self._tokens = tokens
        # enc = tiktoken.get_encoding('gpt2')
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # self.tokens = self.tokens.repeat(25) # Make dataset artiffically bigger
        print(f"Loaded {self._length} tokens")

    def __tokenize_file(self, file_path):
        with open(file_path, 'r') as f:
            text = f.read()
            return self._tokenizer(text, return_tensors="pt")['input_ids'].view(-1)

    def __getitem__(self, index):
        index_with_offset = index - self._shred_offset
        if len(self._sherds):
            if index == 0:
                self._current_shred = 0
                self._shred_offset = 0
                self._tokens = self.__tokenize_file(self._sherds[self._current_shred])
            elif index_with_offset >= self._tokens.shape[0] // self.T:
                self._current_shred += 1
                assert self._current_shred < len(self._sherds), "Out of shreds index."
                self._shred_offset += self._tokens.shape[0] // self.T
                self._tokens = torch.cat([self._tokens[index_with_offset * self.T:], self.__tokenize_file(self._sherds[self._current_shred])])

        # buf = self.tokens[index * self.T : (index + 1) * self.T + 1]
        # x = buf[:-1] # inputs
        # y = buf[1:] # targets
        index -= self._shred_offset
        buf = self._tokens[index * self.T : (index + 1) * self.T]
        return {"input_ids": buf, "labels": buf} # Dont shift labels. Use same approach as HF

    def __len__(self):
        return self._length
        
    def is_classification(self):
        return True
        
        
        
class UnifiedDatasetInterface(Dataset):
    def __init__(self, data, n_ouputs: int, is_classification: bool, batching_fn: Optional[Callable[[Self, list[int]], dict[str, torch.Tensor]]] = None):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.data = data
        self._n_outputs = n_ouputs
        self._is_classification = is_classification
        self._batching_fn = partial(batching_fn, self) if batching_fn else None
        
    def __getitem__(self, index: int):
        if self._batching_fn is not None:
            return index
        element = self.data[index]
        return {"x": element[0], "labels": element[1]}

    def __len__(self):
        return len(self.data)
        
    def is_classification(self):
        return self._is_classification
    
    def n_outputs(self):
        return self._n_outputs

    def get_batching_fn(self) -> Callable[[list[int]], dict[str, torch.Tensor]] | None:
        return self._batching_fn

def create_mnist(used_for_autoencoder: bool, val_split: float = 0.1):
    
    # Do not normalize if used for autoencoder
    if used_for_autoencoder:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.130660], [0.3015041])])
        
    train_val = datasets.MNIST(root='./.mnist_data', train=True, download=True, transform=transform)
    test = datasets.MNIST(root='./.mnist_data', train=False, download=True, transform=transform)
    val_size = round(len(train_val) * val_split)
    train, val = random_split(train_val, [len(train_val) - val_size, val_size])
    is_classification = not used_for_autoencoder
    return UnifiedDatasetInterface(train, 10, is_classification), UnifiedDatasetInterface(val, 10, is_classification), UnifiedDatasetInterface(test, 10, is_classification)
    
    
def create_cifar(cifar_type: int, val_split: float = 0.1):

    if cifar_type == 10:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.247, 0.243, 0.261] 
    elif cifar_type == 100: 
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08),
        transforms.RandomRotation(5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    if cifar_type == 10:
        train_val = datasets.CIFAR10(root='./.cifar_data_10', train=True, download=True, transform=train_transform)
        test = datasets.CIFAR10(root='./.cifar_data_10', train=False, download=True, transform=test_transform)
    elif cifar_type == 100:
        train_val = datasets.CIFAR100(root='./.cifar_data', train=True, download=True, transform=train_transform)
        test = datasets.CIFAR100(root='./.cifar_data', train=False, download=True, transform=test_transform)
    else:
        raise Exception(f"Unssuported cifar type: {cifar_type}. Suported are: 10, 100")
    # val_size = round(len(train_val) * val_split)
    # train, val = random_split(train_val, [len(train_val) - val_size, val_size])
    # return UnifiedDatasetInterface(train, cifar_type, True), UnifiedDatasetInterface(val, cifar_type, True), UnifiedDatasetInterface(test, cifar_type, True)
    # No validation dataset
    return UnifiedDatasetInterface(train_val, cifar_type, True), None, UnifiedDatasetInterface(test, cifar_type, True)
    
    
def create_fasion_mnist(used_for_autoencoder: bool, val_split: float = 0.1):
    
    # Do not normalize if used for autoencoder
    if used_for_autoencoder:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.28604], [0.3204534])])
        
    train_val = datasets.FashionMNIST(root="./.fashion_mnist", train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root="./.fashion_mnist", train=False, download=True, transform=transform)
    val_size = round(len(train_val) * val_split)
    train, val = random_split(train_val, [len(train_val) - val_size, val_size])
    is_classification = not used_for_autoencoder
    return UnifiedDatasetInterface(train, 10, is_classification), UnifiedDatasetInterface(val, 10, is_classification), UnifiedDatasetInterface(test, 10, is_classification)
    
# TODO: Create Test split
def create_sst(tokenizer: AutoTokenizer):
    train_data = load_dataset("nyu-mll/glue", "sst2")['train']
    validation_data = load_dataset("nyu-mll/glue", "sst2")['validation']
    
    def batching(self: UnifiedDatasetInterface, x: list[int]) -> dict[str, torch.Tensor]:
        assert self.tokenizer
        inpts = self.tokenizer([self.data[index]['sentence'] for index in x],  padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inpts['input_ids']
        attention_mask = inpts['attention_mask']
        outputs = torch.tensor([self.data[index]['label'] for index in x])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": outputs}
        
    train_dataset = UnifiedDatasetInterface(train_data, 2, True, batching_fn=batching)
    val_dataset = UnifiedDatasetInterface(validation_data, 2, True, batching_fn=batching)
    train_dataset.tokenizer = tokenizer
    val_dataset.tokenizer = tokenizer
    return train_dataset, val_dataset, None # TODO: Create Test split


def create_qqp(tokenizer: AutoTokenizer):
    train_data = load_dataset("nyu-mll/glue", "qqp")['train']
    validation_data = load_dataset("nyu-mll/glue", "qqp")['validation']
    
    def batching(self: UnifiedDatasetInterface, x: list[int]) -> dict[str, torch.Tensor]:
        assert self.tokenizer
        inpts = self.tokenizer([f"{self.data[index]['question1']}  {self.data[index]['question2']}" for index in x],  padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inpts['input_ids']
        attention_mask = inpts['attention_mask']
        outputs = torch.tensor([self.data[index]['label'] for index in x])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": outputs}
        
    train_dataset = UnifiedDatasetInterface(train_data, 2, True, batching_fn=batching)
    train_dataset.tokenizer = tokenizer
    val_dataset = UnifiedDatasetInterface(validation_data, 2, True, batching_fn=batching)
    val_dataset.tokenizer = tokenizer
    return train_dataset, None, val_dataset # TODO: Create Test split
    # return train_dataset, val_dataset, None # TODO: Create Test split
    
def create_mnli(tokenizer: AutoTokenizer):
    train_data = load_dataset("nyu-mll/glue", "mnli")['train']
    validation_data = load_dataset("nyu-mll/glue", "mnli_matched")['validation']
    # test_data = load_dataset("nyu-mll/glue", "mnli_matched")['test'] # No label in test split
    
    def batching(self: UnifiedDatasetInterface, x: list[int]) -> dict[str, torch.Tensor]:
        assert self.tokenizer
        inpts = self.tokenizer([f"{self.data[index]['premise']}  {self.data[index]['hypothesis']}" for index in x],  padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inpts['input_ids']
        attention_mask = inpts['attention_mask']
        outputs = torch.tensor([self.data[index]['label'] for index in x])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": outputs}
        
    train_dataset = UnifiedDatasetInterface(train_data, 3, True, batching_fn=batching)
    train_dataset.tokenizer = tokenizer
    val_dataset = UnifiedDatasetInterface(validation_data, 3, True, batching_fn=batching)
    val_dataset.tokenizer = tokenizer
    return train_dataset, val_dataset, None
    
def create_imbd(tokenizer: AutoTokenizer):
    dataset = load_dataset("imdb")
    
    def batching(self: UnifiedDatasetInterface, x: list[int]) -> dict[str, torch.Tensor]:
        assert self.tokenizer
        inpts = self.tokenizer([self.data[index]['text'] for index in x],  padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inpts['input_ids']
        attention_mask = inpts['attention_mask']
        outputs = torch.tensor([self.data[index]['label'] for index in x])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": outputs}
        
    train_dataset = UnifiedDatasetInterface(dataset['train'], 2, True, batching_fn=batching)
    train_dataset.tokenizer = tokenizer
    val_dataset = UnifiedDatasetInterface(dataset['train'], 2, True, batching_fn=batching)
    val_dataset.tokenizer = tokenizer
    return train_dataset, val_dataset, None
        
        
        
# class MMLUDataset:
#     def __init__(self, tokenizer) -> None:
#         self.data = load_dataset("lighteval/mmlu", "college_mathematics")['auxiliary_train']
#         self.tokenizer = tokenizer

        
#     def batching(self, x):
#         inpts = self.tokenizer([f"{self.data[index]['question']}  {self.data[index]['choices']}" for index in x],  padding="longest", truncation=True, max_length=512, return_tensors="pt")
#         input_ids = inpts['input_ids']
#         attention_mask = inpts['attention_mask']
#         outputs = torch.tensor([self.data[index]['answer'] for index in x])
#         return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": outputs}
        
def create_boston_datatset(val_split: float = 0.2, seed: int = 42):
    
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, skiprows=22, header=None)
    raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
    X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    y = raw_df.values[1::2, 2]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_split, random_state=seed)
    
    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.get_default_dtype())
    y_train_tensor = torch.tensor(y_train, dtype=torch.get_default_dtype()).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.get_default_dtype())
    y_test_tensor = torch.tensor(y_test, dtype=torch.get_default_dtype()).view(-1, 1)
    
    train_data = list(zip(list(torch.unbind(X_train_tensor, dim=0)), list(torch.unbind(y_train_tensor, dim=0))))
    test_data = list(zip(list(torch.unbind(X_test_tensor, dim=0)), list(torch.unbind(y_test_tensor, dim=0))))

    train_dataset = UnifiedDatasetInterface(train_data, 1, False)
    test_dataset = UnifiedDatasetInterface(test_data, 1, False)
    return train_dataset, None, test_dataset

def create_housing_datatset(val_split: float = 0.2, seed: int = 42):
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_split, random_state=seed)
    # X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_split, random_state=42)
    
    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.get_default_dtype())
    y_train_tensor = torch.tensor(y_train, dtype=torch.get_default_dtype()).view(-1, 1)
    # X_val_tensor = torch.tensor(X_val, dtype=torch.get_default_dtype())
    # y_val_tensor = torch.tensor(y_val, dtype=torch.get_default_dtype()).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.get_default_dtype())
    y_test_tensor = torch.tensor(y_test, dtype=torch.get_default_dtype()).view(-1, 1)
    
    train_data = list(zip(list(torch.unbind(X_train_tensor, dim=0)), list(torch.unbind(y_train_tensor, dim=0))))
    # val_data = list(zip(list(torch.unbind(X_val_tensor, dim=0)), list(torch.unbind(y_val_tensor, dim=0))))
    test_data = list(zip(list(torch.unbind(X_test_tensor, dim=0)), list(torch.unbind(y_test_tensor, dim=0))))

    train_dataset = UnifiedDatasetInterface(train_data, 1, False)
    # val_dataset = UnifiedDatasetInterface(val_data, 1, False)
    test_dataset = UnifiedDatasetInterface(test_data, 1, False)
    return train_dataset, None, test_dataset
    
    
def create_energy_datatset(file=".datasets/energy_efficiency_data.csv"):
    data = pd.read_csv(file)
    scaler = StandardScaler()
    X = scaler.fit_transform(data.iloc[:, :-2].values)
    y = data.iloc[:, -2:].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.get_default_dtype())
    y_train_tensor = torch.tensor(y_train, dtype=torch.get_default_dtype()).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.get_default_dtype())
    y_val_tensor = torch.tensor(y_val, dtype=torch.get_default_dtype()).view(-1, 1)
    
    train_data = list(zip(list(torch.unbind(X_train_tensor, dim=0)), list(torch.unbind(y_train_tensor, dim=0))))
    val_data = list(zip(list(torch.unbind(X_val_tensor, dim=0)), list(torch.unbind(y_val_tensor, dim=0))))

    train_dataset = UnifiedDatasetInterface(train_data, 2, False)
    val_dataset = UnifiedDatasetInterface(val_data, 2, False)
    return train_dataset, val_dataset, None # TODO: Create Test split

