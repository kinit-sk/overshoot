import os
import tiktoken
import torch
from torchvision import datasets, transforms
from datasets import load_dataset


class NextTokenDataloader:
    
    def __init__(self, T: int, shift_labels: bool = True, source_file: str = 'tiny_shakespear.txt', cache_dir='.next-token-dataloader'):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
        self.T = T
        self.shift_labels = shift_labels

        file_path = os.path.join(cache_dir, source_file)
        # Download source file if needed
        if not os.path.exists(file_path):
            download_links = {
                "tiny_shakespear.txt": 'https://drive.google.com/uc?id=1zPB3Y_9mUfKTrywpOX5Jz9j7TwcMkbnC',
                "gutenberg_books.txt": 'https://drive.google.com/uc?id=10N-sj1Nu2dj6XjyBxi43tfqamxb9Rn6t'
            }
            if source_file not in download_links:
                raise Exception(f'Unsupported source file: {source_file}')
            import gdown
            gdown.download(download_links[source_file], file_path, quiet=False)

        with open(file_path, 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        # self.tokens = self.tokens.repeat(25) # Make dataset artiffically bigger
        print(f"Loaded {len(self.tokens)} tokens")

    def __getitem__(self, index):
        buf = self.tokens[index * self.T : (index + 1) * self.T + 1]
        x = buf[:-1] # inputs
        y = buf[1:] # targets
        if self.shift_labels:
            return {"input_ids": x, "labels": y}
        else:
            return {"input_ids": x, "labels": x}

    def __len__(self):
        return len(self.tokens) // (self.T + 1)
        
        
        
        

class Cifar100Dataset:
    def __init__(self) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with mean and std deviation for MNIST
            # transforms.RandomRotation(10),
        ])
        # self.dataset = datasets.MNIST(root='./.mnist_data', train=True, download=True, transform=transform)
        # self.dataset = datasets.MNIST(root='./.mnist_data', train=True, download=True)
        self.dataset = datasets.CIFAR100(root='./.cifar_data', train=True, download=True, transform=transform)
        
    def __getitem__(self, index):
        return {"x": self.dataset[index][0], "labels": self.dataset[index][1]}

    def __len__(self):
        return len(self.dataset)
        
        
        
class SST2Datatset:
    def __init__(self, model_tokenizer: str) -> None:
        data = load_dataset("nyu-mll/glue", "sst2")['train']
        tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)
        inpts = tokenizer([d['sentence'] for d in data],  padding="longest", truncation=True, max_length=512, return_tensors="pt")
        self.input_ids = inpts['input_ids']
        self.attention_mask = inpts['attention_mask']
        self.outputs = [d['label'] for d in data]

    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index], "attention_mask": self.attention_mask[index], "labels": self.outputs[index]}

    def __len__(self):
        return len(self.input_ids)
        
class QQPDataset:
    def __init__(self, tokenizer: str) -> None:
        data = load_dataset("nyu-mll/glue", "qqp")['train']
        inpts = tokenizer([f"{d['question1']}  {d['question2']}" for d in data],  padding="longest", truncation=True, max_length=512, return_tensors="pt")
        self.input_ids = inpts['input_ids']
        self.attention_mask = inpts['attention_mask']
        self.outputs = [d['label'] for d in data]

    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index], "attention_mask": self.attention_mask[index], "labels": self.outputs[index]}

    def __len__(self):
        return len(self.input_ids)
        
class MNLIDataset:
    def __init__(self, tokenizer) -> None:
        data = load_dataset("nyu-mll/glue", "mnli_matched")['validation']
        inpts = tokenizer([f"{d['premise']}  {d['hypothesis']}" for d in data],  padding="longest", truncation=True, max_length=512, return_tensors="pt")
        self.input_ids = inpts['input_ids']
        self.attention_mask = inpts['attention_mask']
        self.outputs = [d['label'] for d in data]

    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index], "attention_mask": self.attention_mask[index], "labels": self.outputs[index]}

    def __len__(self):
        return len(self.input_ids)
        
class MMLUDataset:
    def __init__(self, tokenizer) -> None:
        data = load_dataset("lighteval/mmlu", "college_mathematics")['auxiliary_train']
        inpts = tokenizer([f"{d['question']}  {d['choices']}" for d in data],  padding="longest", truncation=True, max_length=1024, return_tensors="pt")
        self.input_ids = inpts['input_ids']
        self.attention_mask = inpts['attention_mask']
        self.outputs = [d['answer'] for d in data]

    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index], "attention_mask": self.attention_mask[index], "labels": self.outputs[index]}

    def __len__(self):
        return len(self.input_ids)