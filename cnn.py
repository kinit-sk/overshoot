import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CNN(nn.Module):
    def __init__(self, inpt_shape, output_shape):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                # nn.Conv2d(inpt_shape[-1], 32, 3, padding='same'),
                # nn.Conv2d(32, 64, 3, padding='same'),
                # nn.Conv2d(64, 128, 3, padding='same'),
                
                nn.Conv2d(inpt_shape[-1], 32, 3, padding='same'),
                nn.Conv2d(32, 64, 3, padding='same'),
            ]
        )
        self.fc1 = nn.Linear(round(inpt_shape[0] / 2**len(self.convs))**2 * 64, 50)
        self.fc2 = nn.Linear(50, output_shape)
        # self.dropout = nn.Dropout(0.5)
        
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


class ResNet(nn.Module):

    def __init__(self, num_classes: int, type: str):
        super().__init__()
        self._ce_loss = torch.nn.CrossEntropyLoss()
        if type == 'resnet18':
            self._model = models.resnet18(pretrained=False)
        elif type == 'resnet34':
            self._model = models.resnet34(pretrained=False)
        elif type == 'resnet50':
            self._model = models.resnet50(pretrained=False)
        elif type == 'resnet101':
            self._model = models.resnet101(pretrained=False)
        elif type == 'resnet152':
            self._model = models.resnet152(pretrained=False)
        else:
            raise Exception(f'Invalid ResNet type: {type}')
            
        self._model.fc = nn.Linear(self._model.fc.in_features, num_classes)

    def forward(self, x, labels=None):
        logits = self._model(x)
        loss = None
        if labels is not None:
            loss = self._ce_loss(logits, labels)
        return {'loss': loss, 'logits': logits}
