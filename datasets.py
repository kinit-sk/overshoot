import os
import tiktoken
import torch


class NextTokenDataloader:
    
    def __init__(self, T: int, source_file: str = 'tiny_shakespear.txt', cache_dir='.next-token-dataloader'):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
        self.T = T

        file_path = os.path.join(cache_dir, source_file)
        # Download source file if needed
        if not os.path.exists(file_path):
            download_links = {
                "tiny_shakespear.txt": 'https://drive.google.com/uc?id=1zPB3Y_9mUfKTrywpOX5Jz9j7TwcMkbnC'
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
        # self.tokens = self.tokens.repeat(100) # Make dataset artiffically bigger
        print(f"Loaded {len(self.tokens)} tokens")

    def __getitem__(self, index):
        buf = self.tokens[index * self.T : (index + 1) * self.T + 1]
        x = buf[:-1] # inputs
        y = buf[1:] # targets
        return x, y

    def __len__(self):
        return len(self.tokens) // (self.T + 1)