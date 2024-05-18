import torch

# Print the number of available GPUs
print(f"Number of available GPUs: {torch.cuda.device_count()}")

# Check CUDA availability and select the appropriate device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
