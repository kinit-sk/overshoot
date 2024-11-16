import torch
import time
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torch.optim as optim

import random
seed = 422
random.seed(seed)
torch.manual_seed(seed)
torch.set_default_dtype(torch.float64)


            

# Data Augmentation
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
])

# Datasets and Loaders
train_dataset = CIFAR100(root='./.cifar_data', train=True, download=True, transform=transform_train)
test_dataset = CIFAR100(root='./.cifar_data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# Model
model = torchvision.models.resnet18(num_classes=100).cuda()

# Optimizer and Scheduler
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer = optim.AdamW(model.parameters(), weight_decay=0)
# optimizer = optim.SGD(model.parameters(), momentum=0.9, nesterov=True)
# optimizer = optim.SGD(model.parameters(), momentum=0.9)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
criterion = nn.CrossEntropyLoss()

# Training Loop
training_step = 0
for epoch in range(200):  # Training for 200 epochs
    model.train()
    start_time = time.time()
    for inputs, targets in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if training_step % 50 == 0:
            print(f"Step: {training_step}, epoch: {epoch}, loss: {loss.item():.4f}")
        training_step += 1
        # if training_step >= 50:
        #     exit()
        # log_stats(epoch, )
    # scheduler.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    print(f"Epoch took {(time.time() - start_time)} seconds")
    print(f"Epoch {epoch + 1}: Test Accuracy: {100 * correct / total:.2f}%")