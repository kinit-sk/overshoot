from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class _2c2d(nn.Module):
    def __init__(self, inpt_shape: list[int], n_outputs: int):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(inpt_shape[0], 16, 3, padding='same'),
                nn.Conv2d(16, 32, 3, padding='same'),
            ]
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *inpt_shape)
            for conv in self.convs:
                dummy = conv(dummy)
                dummy = F.max_pool2d(dummy, 2)
            n_flatten = dummy.numel() // dummy.size(0)
        self.fc1 = nn.Linear(n_flatten, 64)
        self.fc2 = nn.Linear(64, n_outputs)
        
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor | None]:
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

