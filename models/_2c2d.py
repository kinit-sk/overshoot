import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class _2c2d(nn.Module):
    def __init__(self, inpt_shape, output_shape):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(inpt_shape[0], 32, 3, padding='same'),
                nn.Conv2d(32, 64, 3, padding='same'),
            ]
        )
        self.fc1 = nn.Linear(round(inpt_shape[-1] / 2**len(self.convs))**2 * 64, 256)
        self.fc2 = nn.Linear(256, output_shape)
        
    def forward(self, x, labels=None):
        for conv in self.convs:
            x = conv(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        logits = F.log_softmax(x, dim=1)
        
        loss = None
        if labels is not None:
            loss = F.nll_loss(logits, labels)
        return {'loss': loss, 'logits': logits}

