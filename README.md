# Overshoot: Taking advantage of future gradients in momentum-based stochastic optimization
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

## Try Overshoot on MNIST

1. Install Overshoot optimizer
```
pip install git+https://github.com/kinit-sk/overshoot.git
```
2. Train and eval small MLP on MNIST using AdamW, SGD, AdamO, and SGDO

```python
import torch
from torch.optim import AdamW, SGD
from torchvision import datasets, transforms
from overshoot import AdamO, SGDO

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc2(x)
        
def train(model, optimizer):
    print(optimizer.__class__.__name__)
    torch.manual_seed(42)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.130660], [0.3015041])])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(4):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()

        # Move weights to base variant
        if isinstance(optimizer, (AdamO, SGDO)):
            optimizer.move_to_base() 
           
        model.eval()
        with torch.no_grad():
            correct = 0
            for images, labels in test_loader:
                _, predicted = torch.max(model(images), 1)
                correct += (predicted == labels).sum().item()
        print(f"({epoch+1}/4) Test Accuracy: {100 * correct / len(test_loader.dataset):.2f}%")
        
        # Move weights to overshoot variant
        if isinstance(optimizer, (AdamO, SGDO)):
            optimizer.move_to_overshoot() 
            
# Init four equal models
models = [MLP() for _ in range(4)]
for m in models[1:]:
    m.load_state_dict(models[0].state_dict())

# Adam
train(models[0], AdamW(models[0].parameters()))
train(models[1], AdamO(models[1].parameters(), overshoot=5))

# SGD
train(models[2], SGD(models[2].parameters(), lr=0.01, momentum=0.9))
train(models[3], SGDO(models[3].parameters(), lr=0.01, momentum=0.9, overshoot=5))
```
## Test Overshoot in various scenarios

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


