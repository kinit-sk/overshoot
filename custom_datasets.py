import os
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
        self._batching_fn = None
        
    def __getitem__(self, index):
        return {"x": self.data[index][0], "labels": self.data[index][1]}

    def __len__(self):
        return len(self.data)
        
    def is_classification(self):
        return self._is_classification
    
    def n_outputs(self):
        return self._n_outputs

    def get_batching_fn(self):
        return self._batching_fn

def create_mnist():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), 
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with mean and std deviation for MNIST
    ])
    train = datasets.MNIST(root='./.mnist_data', train=True, download=True, transform=transform)
    val = datasets.MNIST(root='./.mnist_data', train=False, download=True, transform=transform)
    return UnifiedDatasetInterface(train, 10, True), UnifiedDatasetInterface(val, 10, True)
    
    
    
def create_cifar(cifar_type: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        # transforms.RandomRotation(10),
    ])
    if cifar_type == 10:
        train = datasets.CIFAR10(root='./.cifar_data_10', train=True, download=True, transform=transform)
        test = datasets.CIFAR10(root='./.cifar_data_10', train=False, download=True, transform=transform)
    elif cifar_type == 100:
        train = datasets.CIFAR100(root='./.cifar_data', train=True, download=True, transform=transform)
        test = datasets.CIFAR100(root='./.cifar_data', train=False, download=True, transform=transform)
    else:
        raise Exception(f"Unssuported cifar type: {cifar_type}. Suported are: 10, 100")
    return UnifiedDatasetInterface(train, cifar_type, True), UnifiedDatasetInterface(test, cifar_type, True)



class SST2Datatset:
    def __init__(self, tokenizer: str) -> None:
        self.data = load_dataset("nyu-mll/glue", "sst2")['train']
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        return index
        
    def __len__(self):
        return len(self.data)
    
    def batching(self, x):
        inpts = self.tokenizer([self.data[index]['sentence'] for index in x],  padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inpts['input_ids']
        attention_mask = inpts['attention_mask']
        outputs = torch.tensor([self.data[index]['label'] for index in x])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": outputs}
        
    def is_classification(self):
        return True
        
    def n_outputs(self):
        return 2
        
        
class QQPDataset:
    def __init__(self, tokenizer: str) -> None:
        self.data = load_dataset("nyu-mll/glue", "qqp")['train']
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        return index

    def __len__(self):
        return len(self.data)
        
    def batching(self, x):
        inpts = self.tokenizer([f"{self.data[index]['question1']}  {self.data[index]['question2']}" for index in x],  padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inpts['input_ids']
        attention_mask = inpts['attention_mask']
        outputs = torch.tensor([self.data[index]['label'] for index in x])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": outputs}
        
    def is_classification(self):
        return True
        
    def n_outputs(self):
        return 2
        
class MNLIDataset:
    def __init__(self, tokenizer: str) -> None:
        self.data = load_dataset("nyu-mll/glue", "mnli_matched")['validation']
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        return index

    def __len__(self):
        return len(self.data)
        
    def batching(self, x):
        inpts = self.tokenizer([f"{self.data[index]['premise']}  {self.data[index]['hypothesis']}" for index in x],  padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inpts['input_ids']
        attention_mask = inpts['attention_mask']
        outputs = torch.tensor([self.data[index]['label'] for index in x])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": outputs}
        
    def is_classification(self):
        return True
        
    def n_outputs(self):
        return 3
        
class MMLUDataset:
    def __init__(self, tokenizer) -> None:
        self.data = load_dataset("lighteval/mmlu", "college_mathematics")['auxiliary_train']
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        return index

    def __len__(self):
        return len(self.data)
        
    def batching(self, x):
        inpts = self.tokenizer([f"{self.data[index]['question']}  {self.data[index]['choices']}" for index in x],  padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inpts['input_ids']
        attention_mask = inpts['attention_mask']
        outputs = torch.tensor([self.data[index]['answer'] for index in x])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": outputs}
        
    def is_classification(self):
        return True
        
    # TODO: Check if this is correct
    def n_outputs(self):
        return 4


def create_housing_datatset():
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.get_default_dtype())
    y_train_tensor = torch.tensor(y_train, dtype=torch.get_default_dtype()).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.get_default_dtype())
    y_val_tensor = torch.tensor(y_val, dtype=torch.get_default_dtype()).view(-1, 1)
    
    train_data = list(zip(list(torch.unbind(X_train_tensor, dim=0)), list(torch.unbind(y_train_tensor, dim=0))))
    val_data = list(zip(list(torch.unbind(X_val_tensor, dim=0)), list(torch.unbind(y_val_tensor, dim=0))))

    train_dataset = UnifiedDatasetInterface(train_data, 1, False)
    val_dataset = UnifiedDatasetInterface(val_data, 1, False)
    return train_dataset, val_dataset

    
class CaliforniaHousingDataset:

    def __init__(self):
        dataset = fetch_california_housing()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(dataset.data)
        self.X = torch.tensor(X_train, dtype=torch.float32)
        self.labels = torch.tensor(dataset.target, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self, index):
        return {"x": self.X[index], "labels": self.labels[index]}
        
    def is_classification(self):
        return False
        
    def n_outputs(self):
        return 1
        
        
class EnergyDataset:

    def __init__(self, file=".datasets/energy_efficiency_data.csv"):
        data = pd.read_csv(file)
        scaler = StandardScaler()
        self.X = torch.tensor(scaler.fit_transform(data.iloc[:, :-2].values), dtype=torch.float32)
        self.labels = torch.tensor(data.iloc[:, -2:].values, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self, index):
        return {"x": self.X[index], "labels": self.labels[index]}

    def is_classification(self):
        return False
        
    def n_outputs(self):
        return 2