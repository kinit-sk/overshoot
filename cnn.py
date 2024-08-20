import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
    def forward(self, x, labels=None):
        for conv in self.convs:
            x = conv(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        logits = F.log_softmax(x, dim=1)
        loss = None
        if labels is not None:
            loss = F.nll_loss(logits, labels)
        return {'loss': loss, 'logits': logits}
