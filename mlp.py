import math

import torch
import torch.nn as nn
from typing import Sequence

class MLP(nn.Module):
    def __init__(self, inpt_shape, output_size: int, is_classification: bool, hidden_layers: Sequence[int] = [50]):
        super().__init__()

        sizes = [math.prod(inpt_shape)] + hidden_layers + [output_size]
        self.layers = nn.ModuleList(
            [nn.Linear(before, after) for before, after in zip(sizes[:-1], sizes[1:])]
        )
        self.relu = nn.ReLU()
        self.loss_fn = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()
        
    def forward(self, x, labels = None):
        x = torch.flatten(x, 1)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.relu(x)
        logits = self.layers[-1](x)
        loss = None if labels is None else self.loss_fn(logits, labels)
        return {'loss': loss, 'logits': logits}
