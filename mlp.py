import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, inpt_shape, output_shape, hidden_layers = [128, 50]):
        super().__init__()

        sizes = [math.prod(inpt_shape)] + hidden_layers + [output_shape]
        self.layers = nn.ModuleList(
            [nn.Linear(before, after) for before, after in zip(sizes[:-1], sizes[1:])]
        )
        self.relu = nn.ReLU()
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x, labels = None):
        x = torch.flatten(x, 1)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.relu(x)
        logits = self.layers[-1](x)
        loss = None if labels is None else self.loss_fn(logits, labels)
        return {'loss': loss, 'logits': logits}
