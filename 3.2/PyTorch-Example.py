"""
PROGRAM: PyTorch GPU Check and Minimal Inference Demo
AUTHOR:  Dr. Aliasghar Khavasi
DATE:    2026 - January

DESCRIPTION:
  A short PyTorch script that:
    1) Checks whether CUDA is available and selects the compute device (GPU if available, otherwise CPU).
    2) Defines a small neural network (SimpleModel) with two fully connected layers:
       - Linear(20 → 32) with ReLU activation
       - Linear(32 → 2) followed by Softmax to produce class probabilities
    3) Instantiates the model on the selected device and prints the model structure.
    4) Generates a dummy input batch (10 samples × 20 features), converts it to a torch tensor,
       and moves it to the selected device.
    5) Runs a forward pass under no_grad() (inference mode) and prints the predicted probability vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Check GPU availability
print("CUDA Available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(20, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

model = SimpleModel().to(device)
print(model)  # Print model summary-like info

# Generate dummy data
data = np.random.rand(10, 20).astype(np.float32)  # 10 samples, 20 features each
data_torch = torch.from_numpy(data).to(device)

# Forward pass
with torch.no_grad():  # Disable gradient calculations for this test
    predictions = model(data_torch)

print("Predictions:", predictions)
