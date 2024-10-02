import os
import tiktoken
import torch
from torchvision import datasets, transforms
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

        if source_file.endswith("_"):
            self._sherds = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.startswith(source_file)]
        else:
            self._sherds = [file_path]
            
        self._current_shred = 0
        self._shred_offset = 0
        self._length = 0
        for i, s in enumerate(self._sherds):
            tokens = self.__tokenize_file(s)
            self._length += tokens.shape[0] // self.T
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
        
        if len(self._sherds):
            if index == 0:
                self._current_shred = 0
                self._shred_offset = 0
                self._tokens = self.__tokenize_file(self._sherds[self._current_shred])
            elif index - self._shred_offset >= self._tokens.shape[0] // self.T:
                self._current_shred += 1
                assert self._current_shred < len(self._sherds), "Reached end of sherd list"
                self._shred_offset += self._tokens.shape[0] // self.T
                self._tokens = self.__tokenize_file(self._sherds[self._current_shred])

        # buf = self.tokens[index * self.T : (index + 1) * self.T + 1]
        # x = buf[:-1] # inputs
        # y = buf[1:] # targets
        index -= self._shred_offset
        buf = self._tokens[index * self.T : (index + 1) * self.T]
        return {"input_ids": buf, "labels": buf} # Dont shift labels. Use same approach as HF

    def __len__(self):
        return self._length
        
        
        
class MnistDataset:
    def __init__(self) -> None:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3), 
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with mean and std deviation for MNIST
            # transforms.RandomRotation(10),
        ])
        self.dataset = datasets.MNIST(root='./.mnist_data', train=True, download=True, transform=transform)
        
    def __getitem__(self, index):
        return {"x": self.dataset[index][0], "labels": self.dataset[index][1]}

    def __len__(self):
        return len(self.dataset)
        

class Cifar100Dataset:
    def __init__(self) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with mean and std deviation for MNIST
            # transforms.RandomRotation(10),
        ])
        self.dataset = datasets.CIFAR100(root='./.cifar_data', train=True, download=True, transform=transform)
        
    def __getitem__(self, index):
        return {"x": self.dataset[index][0], "labels": self.dataset[index][1]}

    def __len__(self):
        return len(self.dataset)
        
class Cifar10Dataset:
    def __init__(self) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with mean and std deviation for MNIST
            # transforms.RandomRotation(10),
        ])
        self.dataset = datasets.CIFAR10(root='./.cifar_data_10', train=True, download=True, transform=transform)
        
    def __getitem__(self, index):
        return {"x": self.dataset[index][0], "labels": self.dataset[index][1]}

    def __len__(self):
        return len(self.dataset)
        
        

class SST2Datatset:
    def __init__(self, tokenizer: str) -> None:
        self.data = load_dataset("nyu-mll/glue", "sst2")['train']
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        inpts = self.tokenizer([self.data[index]['sentence']],  padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inpts['input_ids'][0]
        attention_mask = inpts['attention_mask'][0]
        outputs = self.data[index]['label']
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": outputs}

    def __len__(self):
        return len(self.data)
        
        
        
class QQPDataset:
    def __init__(self, tokenizer: str) -> None:
        self.data = load_dataset("nyu-mll/glue", "qqp")['train']
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        inpts = self.tokenizer([f"{self.data[index]['question1']}  {self.data[index]['question2']}"],  padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inpts['input_ids'][0]
        attention_mask = inpts['attention_mask'][0]
        outputs = self.data[index]['label']
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": outputs}

    def __len__(self):
        return len(self.data)
        
class MNLIDataset:
    def __init__(self, tokenizer: str) -> None:
        self.data = load_dataset("nyu-mll/glue", "mnli_matched")['validation']
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        inpts = self.tokenizer([f"{self.data[index]['premise']}  {self.data[index]['hypothesis']}"],  padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inpts['input_ids'][0]
        attention_mask = inpts['attention_mask'][0]
        outputs = self.data[index]['label']
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": outputs}

    def __len__(self):
        return len(self.data)
        
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