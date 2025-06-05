---
layout: post
title: "Getting Started with Deep Learning using PyTorch"
date: 2025-01-10 14:30:00 +0530
categories: [dl]
tags:
  - deep-learning
  - pytorch
  - neural-networks
  - tutorial
---

PyTorch has become one of the most popular frameworks for deep learning research and production. In this tutorial, we'll build your first neural network from scratch and understand the core concepts.

## Why PyTorch?

PyTorch offers several advantages for deep learning:

- **Dynamic Computation Graphs**: More intuitive debugging
- **Pythonic**: Feels natural to Python developers  
- **Strong Community**: Excellent documentation and tutorials
- **Research-Friendly**: Easy to experiment with new ideas

## Setting Up Your Environment

```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# For GPU support (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Building Your First Neural Network

Let's create a simple neural network for image classification:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define the neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize the model
model = SimpleNet(input_size=28*28, hidden_size=128, num_classes=10)
```

## Loading and Preprocessing Data

```python
# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

## Training the Model

```python
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Step [{batch_idx}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/len(train_loader):.4f}')
```

## Evaluating the Model

```python
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# Evaluate the model
accuracy = evaluate_model(model, test_loader, device)
```

## Key PyTorch Concepts

### 1. Tensors
PyTorch tensors are similar to NumPy arrays but with GPU support:

```python
# Creating tensors
x = torch.tensor([1, 2, 3])
y = torch.randn(3, 4)
z = torch.zeros(2, 3, dtype=torch.float32)

# Moving to GPU
if torch.cuda.is_available():
    x = x.cuda()
```

### 2. Autograd
Automatic differentiation for computing gradients:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # Prints tensor([4.])
```

### 3. nn.Module
Base class for all neural network modules:

```python
class CustomLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        return torch.relu(self.linear(x))
```

## Advanced Tips

### 1. Learning Rate Scheduling
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
```

### 2. Model Checkpointing
```python
# Save model
torch.save(model.state_dict(), 'model_checkpoint.pth')

# Load model
model.load_state_dict(torch.load('model_checkpoint.pth'))
```

### 3. Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, targets in train_loader:
    with autocast():
        outputs = model(data)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Common Pitfalls to Avoid

1. **Forgetting to call `model.train()` and `model.eval()`**
2. **Not moving data to the same device as the model**
3. **Forgetting to zero gradients with `optimizer.zero_grad()`**
4. **Using the wrong loss function for your task**
5. **Not normalizing your input data**

## Next Steps

Now that you understand the basics, explore these advanced topics:

- **Convolutional Neural Networks (CNNs)** for image processing
- **Recurrent Neural Networks (RNNs)** for sequence data
- **Transfer Learning** with pre-trained models
- **Custom Datasets** and data loading
- **Distributed Training** for large-scale models

## Conclusion

PyTorch provides an intuitive and flexible framework for deep learning. Start with simple projects, understand the fundamentals, and gradually work your way up to more complex architectures.

The key to mastering PyTorch is practice—build projects, experiment with different architectures, and don't be afraid to dive into the documentation!

---

*Have you tried PyTorch for your deep learning projects? What challenges did you face? Share your experience in the comments!* 