import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CNNTrainerConfig:
    n_gpu: int = torch.cuda.device_count()  # Use all available gpus
    B: int = 8
    lr_base: float = 3e-3
    lr_overshoot: Optional[None] = None
    epochs: int = 1
    adam_betas: Tuple[float, float] = 0.9, 0.95
    weight_decay: float = 0.1
    accumulate_grad_batches: int = 2
    gradient_clip_val: float = 0.5

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 10)
        # self.fc2 = nn.Linear(128, 10)
        
        
    def forward(self, x, targets=None):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        loss = None
        if targets is not None:
            loss = F.nll_loss(output, targets)
        return output, loss