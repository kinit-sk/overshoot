import os
from functools import partial
from typing import Callable, Optional

import tiktoken
import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import random_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datasets import load_dataset


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
        
        
        
class UnifiedDatasetInterface:
    def __init__(self, data, n_ouputs: int, is_classification: bool, batching_fn: Optional[Callable] = None):
        self.data = data
        self._n_outputs = n_ouputs
        self._is_classification = is_classification
        self._batching_fn = batching_fn
        if self._batching_fn is not None:
            self._batching_fn = partial(self._batching_fn, self)
        
    def __getitem__(self, index):
        if self._batching_fn is not None:
            return index
        return {"x": self.data[index][0], "labels": self.data[index][1]}

    def __len__(self):
        return len(self.data)
        
    def is_classification(self):
        return self._is_classification
    
    def n_outputs(self):
        return self._n_outputs

    def get_batching_fn(self):
        return self._batching_fn

def create_mnist(val_split: float = 0.1):
    transform = transforms.Compose([transforms.ToTensor()])
    train_val = datasets.MNIST(root='./.mnist_data', train=True, download=True, transform=transform)
    test = datasets.MNIST(root='./.mnist_data', train=False, download=True, transform=transform)
    val_size = round(len(train_val) * val_split)
    train, val = random_split(train_val, [len(train_val) - val_size, val_size])
    return UnifiedDatasetInterface(train, 10, True), UnifiedDatasetInterface(val, 10, True), UnifiedDatasetInterface(test, 10, True)
    
    
def create_cifar(cifar_type: int, val_split: float = 0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),
        # transforms.RandomRotation(10),
    ])
    if cifar_type == 10:
        train_val = datasets.CIFAR10(root='./.cifar_data_10', train=True, download=True, transform=transform)
        test = datasets.CIFAR10(root='./.cifar_data_10', train=False, download=True, transform=transform)
    elif cifar_type == 100:
        train_val = datasets.CIFAR100(root='./.cifar_data', train=True, download=True, transform=transform)
        test = datasets.CIFAR100(root='./.cifar_data', train=False, download=True, transform=transform)
    else:
        raise Exception(f"Unssuported cifar type: {cifar_type}. Suported are: 10, 100")
    val_size = round(len(train_val) * val_split)
    train, val = random_split(train_val, [len(train_val) - val_size, val_size])
    return UnifiedDatasetInterface(train, cifar_type, True), UnifiedDatasetInterface(val, cifar_type, True), UnifiedDatasetInterface(test, cifar_type, True)
    
    
def create_fasion_mnist(val_split: float = 0.1):
    transform = transforms.Compose([transforms.ToTensor()])
    train_val = datasets.FashionMNIST(root="./.fashion_mnist", train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root="./.fashion_mnist", train=False, download=True, transform=transform)
    val_size = round(len(train_val) * val_split)
    train, val = random_split(train_val, [len(train_val) - val_size, val_size])
    return UnifiedDatasetInterface(train, 1, False), UnifiedDatasetInterface(val, 1, False), UnifiedDatasetInterface(test, 1, False)
    
# TODO: Create Test split
def create_sst(tokenizer):
    train_data = load_dataset("nyu-mll/glue", "sst2")['train']
    validation_data = load_dataset("nyu-mll/glue", "sst2")['validation']
    
    def batching(self, x):
        inpts = self.tokenizer([self.data[index]['sentence'] for index in x],  padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inpts['input_ids']
        attention_mask = inpts['attention_mask']
        outputs = torch.tensor([self.data[index]['label'] for index in x])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": outputs}
        
    train_dataset = UnifiedDatasetInterface(train_data, 2, True, batching_fn=batching)
    train_dataset.tokenizer = tokenizer
    val_dataset =UnifiedDatasetInterface(validation_data, 2, True, batching_fn=batching)
    val_dataset.tokenizer = tokenizer
    return train_dataset, val_dataset, None # TODO: Create Test split


def create_qqp(tokenizer):
    train_data = load_dataset("nyu-mll/glue", "qqp")['train']
    validation_data = load_dataset("nyu-mll/glue", "qqp")['train']
    
    def batching(self, x):
        inpts = self.tokenizer([f"{self.data[index]['question1']}  {self.data[index]['question2']}" for index in x],  padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inpts['input_ids']
        attention_mask = inpts['attention_mask']
        outputs = torch.tensor([self.data[index]['label'] for index in x])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": outputs}
        
    train_dataset = UnifiedDatasetInterface(train_data, 2, True, batching_fn=batching)
    train_dataset.tokenizer = tokenizer
    val_dataset =UnifiedDatasetInterface(validation_data, 2, True, batching_fn=batching)
    val_dataset.tokenizer = tokenizer
    return train_dataset, val_dataset, None # TODO: Create Test split
        
        
# class MNLIDataset:
#     def __init__(self, tokenizer: str) -> None:
#         self.data = load_dataset("nyu-mll/glue", "mnli_matched")['validation']
#         self.tokenizer = tokenizer

#     def __getitem__(self, index):
#         return index

#     def __len__(self):
#         return len(self.data)
        
#     def batching(self, x):
#         inpts = self.tokenizer([f"{self.data[index]['premise']}  {self.data[index]['hypothesis']}" for index in x],  padding="longest", truncation=True, max_length=512, return_tensors="pt")
#         input_ids = inpts['input_ids']
#         attention_mask = inpts['attention_mask']
#         outputs = torch.tensor([self.data[index]['label'] for index in x])
#         return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": outputs}
        
#     def is_classification(self):
#         return True
        
#     def n_outputs(self):
#         return 3
        
# class MMLUDataset:
#     def __init__(self, tokenizer) -> None:
#         self.data = load_dataset("lighteval/mmlu", "college_mathematics")['auxiliary_train']
#         self.tokenizer = tokenizer

#     def __getitem__(self, index):
#         return index

#     def __len__(self):
#         return len(self.data)
        
#     def batching(self, x):
#         inpts = self.tokenizer([f"{self.data[index]['question']}  {self.data[index]['choices']}" for index in x],  padding="longest", truncation=True, max_length=512, return_tensors="pt")
#         input_ids = inpts['input_ids']
#         attention_mask = inpts['attention_mask']
#         outputs = torch.tensor([self.data[index]['answer'] for index in x])
#         return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": outputs}
        
#     def is_classification(self):
#         return True
        
#     # TODO: Check if this is correct
#     def n_outputs(self):
#         return 4


def create_housing_datatset(val_split: float = 0.1):
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_split, random_state=42)
    
    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.get_default_dtype())
    y_train_tensor = torch.tensor(y_train, dtype=torch.get_default_dtype()).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.get_default_dtype())
    y_val_tensor = torch.tensor(y_val, dtype=torch.get_default_dtype()).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.get_default_dtype())
    y_test_tensor = torch.tensor(y_test, dtype=torch.get_default_dtype()).view(-1, 1)
    
    train_data = list(zip(list(torch.unbind(X_train_tensor, dim=0)), list(torch.unbind(y_train_tensor, dim=0))))
    val_data = list(zip(list(torch.unbind(X_val_tensor, dim=0)), list(torch.unbind(y_val_tensor, dim=0))))
    test_data = list(zip(list(torch.unbind(X_test_tensor, dim=0)), list(torch.unbind(y_test_tensor, dim=0))))

    train_dataset = UnifiedDatasetInterface(train_data, 1, False)
    val_dataset = UnifiedDatasetInterface(val_data, 1, False)
    test_dataset = UnifiedDatasetInterface(test_data, 1, False)
    return train_dataset, val_dataset, test_dataset
    
    
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

