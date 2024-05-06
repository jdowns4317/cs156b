import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple neural network model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda:10.2")  # Use the first GPU
    print("CUDA is available. Running on GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Running on CPU.")

# Create a simple model instance
model = SimpleNet().to(device)  

# Create a dummy input tensor
input_tensor = torch.randn(1, 10).to(device)  

# Perform a forward pass
output = model(input_tensor)
print("Output:", output)
