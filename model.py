import os
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Custom Dataset class
class ImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, test=False):
        """
        Args:
            dataframe (DataFrame): DataFrame with image paths and optionally labels.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            test (bool, optional): If true, does not load labels.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = self.dataframe.iloc[idx, 1]
            return image, label

# Transformation
transform = transforms.Compose([
    transforms.Resize((30, 30)),  # Resize the image
    transforms.ToTensor()         # Convert images to PyTorch tensors
])

# Load CSVs
train_df = pd.read_csv('../../../data/student_labels/train2023.csv')
test_df = pd.read_csv('../../../data/student_labels/test_ids.csv')

# Initialize Dataset and DataLoader for training
train_dataset = ImageDataset(dataframe=train_df, root_dir='../../../data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Adjust num_workers based on your system

# Neural Network Model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(900, 20),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(20, 9)  # Adjusted output size for multi-label classification
)

# Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training Function
def train_nn(train_loader):
    model.train()
    for epoch in range(10):  # Number of epochs
        for data, target in train_loader:
            optimizer.zero_grad()     # Zero the gradients
            output = model(data)      # Forward pass
            loss = loss_fn(output, target)  # Compute loss
            loss.backward()           # Backward pass
            optimizer.step()          # Update weights
    return model

# Train the model
model = train_nn(train_loader)

# Function to get predictions
def get_output(model, test_loader):
    model.eval()
    test_preds = []
    with torch.no_grad():
        for data in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_preds.extend(pred.flatten().tolist())
    return test_preds

# Initialize Dataset and DataLoader for testing
test_dataset = ImageDataset(dataframe=test_df, root_dir='../../../data', transform=transform, test=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Get predictions
test_predictions = get_output(model, test_loader)

# Output results
test_df['Predictions'] = test_predictions
test_df.to_csv('predictions.csv', index=False)
