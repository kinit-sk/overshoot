import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, 64, 3, 1),
                nn.Conv2d(64, 64, 3, 1),
                nn.Conv2d(64, 128, 3, 1),
            ]
        )
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x, target):
        for conv in self.convs:
            x = conv(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        logits = F.log_softmax(x, dim=1)
        return F.nll_loss(logits, target)


class MRELoop(pl.LightningModule):
    def __init__(self):
        super(MRELoop, self).__init__()
        self.model = CNN()
        self.dataset = datasets.MNIST(root=".mnist_data", download=True, transform=transforms.ToTensor())
        self.previous_params = None

    def training_step(self, batch, batch_idx):
        # Check whether new model weights differs from previous ones
        params = torch.cat([param.view(-1) for param in self.model.parameters()])
        if self.previous_params is not None:
            num_different_values = (self.previous_params != params).sum().item()
            self.trainer.should_stop = num_different_values == 0
        else:
            num_different_values = None

        self.previous_params = params
        loss = self.model.forward(*batch)
        print(
            f"step {batch_idx} | diff weights: {num_different_values} | all weights: {params.numel()} | weights mean: {torch.mean(params)} | loss: {loss.item()}"
        )
        return loss

    def configure_optimizers(self):
        # Bug occurs also with different lr only at differnt training step
        return torch.optim.AdamW(self.parameters(), lr=2e-3)
        # return torch.optim.SGD(self.parameters(), lr=9e-4) # Also with SGD

    def train_dataloader(self):
        return DataLoader(self.dataset)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(1337)
    pl_trainer = pl.Trainer(
        precision="16-mixed",  # So far bug has occured only with 16-mixed
        deterministic=True,
        enable_progress_bar=False,
    )
    pl_trainer.fit(MRELoop())
