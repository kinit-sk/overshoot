import math
from typing import Optional

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, inpt_shape: list[int], output_size: int, is_classification: bool, hidden_layers: list[int]):
        super().__init__()

        sizes = [math.prod(inpt_shape)] + hidden_layers + [output_size]
        self.layers = nn.ModuleList(
            [nn.Linear(before, after) for before, after in zip(sizes[:-1], sizes[1:])]
        )
        self.relu = nn.ReLU()
        self.loss_fn = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()
        
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        x = torch.flatten(x, 1)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.relu(x)
        logits = self.layers[-1](x)
        loss = None if labels is None else self.loss_fn(logits, labels)
        return {'loss': loss, 'logits': logits}
