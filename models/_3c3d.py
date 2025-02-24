from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class _3c3d(nn.Module):
    def __init__(self, inpt_shape: list[int], output_shape: list[int]):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(inpt_shape[0], 64, 5, padding='valid'),
                nn.Conv2d(64, 96, 3, padding='valid'),
                nn.Conv2d(96, 128, 3, padding='same'),
            ]
        )
        self.fc1 = nn.Linear(3 * 3 * 128, 512) # TODO: This is input shape dependent
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_shape)
        
    def forward(self, x: torch.Tensor, labels:Optional[torch.Tensor] = None):
        for conv in self.convs:
            x = conv(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        logits = F.log_softmax(x, dim=1)
        
        loss = None
        if labels is not None:
            loss = F.nll_loss(logits, labels)
        return {'loss': loss, 'logits': logits}

