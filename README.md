# Overshoot: Taking advantage of future gradients in momentum-based stochastic optimization

![Python Versions](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-brightgreen)](https://pytorch.org/get-started/previous-versions/)

This is an official PyTorch implementation of **Overshoot**. See the [paper](https://arxiv.org/abs/2501.09556).

```
@misc{kopal2025overshoottakingadvantagefuture,
      title={Overshoot: Taking advantage of future gradients in momentum-based stochastic optimization}, 
      author={Jakub Kopal and Michal Gregor and Santiago de Leon-Martinez and Jakub Simko},
      year={2025},
      eprint={2501.09556},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.09556}, 
}
```

## Overshoot demo on mnist

1. Install Overshoot optimizer
```
pip install git+https://github.com/kinit-sk/overshoot.git
```
2. Train and eval on mnist using AdamW vs AdamO (AdamW + Overshoot)

```python
import torch
from torchvision import datasets, transforms
from overshoot import AdamO
torch.manual_seed(21)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def train_test(model, optimizer):
    criterion = torch.nn.CrossEntropyLoss()
    # Training loop
    for epoch in range(5):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{5}], Loss: {loss.item():.4f}")

    # Move weights to base variant
    if isinstance(optimizer, AdamO):
        optimizer.move_to_base() 
        
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

print("AdamW")
model = CNN().to(device)
train_test(model, torch.optim.AdamW(model.parameters()))

print("AdamO")
model = CNN().to(device)
train_test(model, AdamO(model.parameters()))
```
## Benchmark Overshoot on various tasks

### Requirements

 - Python packages: `pip install -r requirements.txt`
 - (Optional) Enviroment with GPU and cuda drivers

### Run

To run baseline:
```
python main.py --model mlp --dataset mnist --opt_name sgd_nesterov
```
To run overshoot with two models implementation:
```
python main.py --model mlp --dataset mnist --opt_name sgd_momentum --two_models --overshoot_factor 0.9
```
To run overshoot with efficient implementation:
```
python main.py --model mlp --dataset mnist --opt_name sgd_overshoot --overshoot_factor 0.9
```
To observe the same results include: `--seed 42 --config_override precision=high`.

For detailed description of the args training entry-point run:
```
python main.py --help
```

### Monitor experiments
To observe training statistics when neither `experiment_name` nor `job_name` is specified run:
```
tensorboard --logdir lightning_logs/test/test --port 6006
```
In the browser open `localhost:6006`.


