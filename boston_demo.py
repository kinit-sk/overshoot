import torch
import torch.nn as nn
import pandas as pd
import numpy as np 
import torch.optim as optim
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(32)

# # Load the dataset
# boston = load_boston()
# X, y = boston.data, boston.target


class UnifiedDatasetInterface:
    def __init__(self, data, n_ouputs: int, is_classification: bool, batching_fn = None):
        self.data = data
        self._n_outputs = n_ouputs
        self._is_classification = is_classification
        self._batching_fn = batching_fn
        
    def __getitem__(self, index):
        if self._batching_fn is not None:
            return index
        element = self.data[index]
        return {"x": element[0], "labels": element[1]}

    def __len__(self):
        return len(self.data)
        
    def is_classification(self):
        return self._is_classification
    
    def n_outputs(self):
        return self._n_outputs

    def get_batching_fn(self):
        return self._batching_fn

def create_housing_datatset(val_split: float = 0.1):
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, skiprows=22, header=None)
    raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
    X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    y = raw_df.values[1::2, 2]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_split, random_state=42)
    
    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.get_default_dtype())
    y_train_tensor = torch.tensor(y_train, dtype=torch.get_default_dtype()).view(-1, 1)
    # X_val_tensor = torch.tensor(X_val, dtype=torch.get_default_dtype())
    # y_val_tensor = torch.tensor(y_val, dtype=torch.get_default_dtype()).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.get_default_dtype())
    y_test_tensor = torch.tensor(y_test, dtype=torch.get_default_dtype()).view(-1, 1)
    
    train_data = list(zip(list(torch.unbind(X_train_tensor, dim=0)), list(torch.unbind(y_train_tensor, dim=0))))
    # val_data = list(zip(list(torch.unbind(X_val_tensor, dim=0)), list(torch.unbind(y_val_tensor, dim=0))))
    test_data = list(zip(list(torch.unbind(X_test_tensor, dim=0)), list(torch.unbind(y_test_tensor, dim=0))))

    train_dataset = UnifiedDatasetInterface(train_data, 1, False)
    # val_dataset = UnifiedDatasetInterface(val_data, 1, False)
    test_dataset = UnifiedDatasetInterface(test_data, 1, False)
    return train_dataset, None, test_dataset






# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, skiprows=22, header=None)
# raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
# X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# y = raw_df.values[1::2, 2]


# # Train-test split
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Normalize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)

# # Convert to PyTorch tensors
# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
# X_val = torch.tensor(X_val, dtype=torch.float32)
# y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# # Create DataLoader
# train_dataset = TensorDataset(X_train, y_train)
# val_dataset = TensorDataset(X_val, y_val)

train_dataset, val_dataset, test_dataset = create_housing_datatset()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=32)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(13, 128),  # Input: 13 features, Hidden Layer: 128 units
            nn.ReLU(),
            nn.Linear(128, 64),  # Hidden Layer: 64 units
            nn.ReLU(),
            nn.Linear(64, 32),   # Hidden Layer: 32 units
            nn.ReLU(),
            nn.Linear(32, 1)     # Output: Single value (regression)
        )
        
    def forward(self, x):
        return self.model(x)

# Initialize model, loss, and optimizer
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_id, batch in enumerate(train_loader):
        # print(torch.mean(batch['x']))
        # print(batch_id, torch.sum(torch.cat([p.data.view(-1).cpu() for p in model.parameters()])))
        # if batch_id > 5:
        #     break
        optimizer.zero_grad()
        y_pred = model(batch['x'])
        loss = criterion(y_pred, batch['labels'])
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch['x'].size(0)
    
    train_loss /= len(train_loader.dataset)

    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            y_pred = model(batch['x'])
            loss = criterion(y_pred, batch['labels'])
            val_loss += loss.item() * batch['x'].size(0)
    
    val_loss /= len(val_loader.dataset)
    
    # Save the best model
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     torch.save(model.state_dict(), "best_mlp_model.pth")
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

print(f"Best Validation Loss: {best_val_loss:.4f}")

# Load the best model for evaluation
model.load_state_dict(torch.load("best_mlp_model.pth"))