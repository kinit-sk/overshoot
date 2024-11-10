import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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
