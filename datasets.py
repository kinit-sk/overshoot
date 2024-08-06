import os
import tiktoken
import torch

os.environ["TIKTOKEN_CACHE_DIR"] = "."

class DataLoaderLite:
    
    def __init__(self, T):
        self.T = T
        with open('gutenberg_books.txt', 'r') as f:
        # with open('tiny_shakespear.txt', 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        # self.tokens = self.tokens.repeat(100) # Make dataset artiffically bigger
        print(f"loaded {len(self.tokens)} tokens")

    def __getitem__(self, index):
        buf = self.tokens[index * self.T : (index + 1) * self.T + 1]
        x = buf[:-1] # inputs
        y = buf[1:] # targets
        return x, y

    def __len__(self):
        return len(self.tokens) // (self.T + 1)