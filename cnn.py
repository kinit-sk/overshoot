import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(3, 32, 3, 1),
                nn.Conv2d(32, 64, 3, 1),
                nn.Conv2d(64, 128, 3, 1),
            ]
        )
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 100)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, labels=None):
        for conv in self.convs:
            x = conv(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        logits = F.log_softmax(x, dim=1)
        
        loss = None
        if labels is not None:
            loss = F.nll_loss(logits, labels)
        return {'loss': loss, 'logits': logits}
