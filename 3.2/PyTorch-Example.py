import torch
import torch.nn as nn
import torch.nn.functional as F

# Force PyTorch to use CPU
device = torch.device("cpu")
print(f"Using device: {device}")
print("PyTorch version:", torch.__version__)


# 1. Define a simple model architecture
class MinimalNet(nn.Module):
    def __init__(self):
        super(MinimalNet, self).__init__()
        self.fc1 = nn.Linear(20, 32)  # Input layer to hidden
        self.fc2 = nn.Linear(32, 2)  # Hidden to output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


# 2. Initialize the model and move to CPU
model = MinimalNet().to(device)
print("\nModel Architecture:\n", model)

# 3. Generate dummy data (10 samples, 20 features)
# PyTorch requires data to be in Tensor format
data = torch.randn(10, 20).to(device)

# 4. Run prediction (forward pass)
model.eval()  # Set to evaluation mode
with torch.no_grad():
    predictions = model(data)

print("\nPredictions shape:", predictions.shape)
print("Predictions:\n", predictions)
