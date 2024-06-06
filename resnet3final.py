import os
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import cv2
import numpy as np

features = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", 
            "Lung Opacity", "Pneumonia", "Pleural Effusion", "Pleural Other",
            "Fracture", "Support Devices"]

feature = sys.argv[1]

bs = 64
num_epochs = 3
w = 256
h = 256
nw = 4

# Check CUDA availability and select the appropriate device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx]['Path'])
        image = cv2.imread(img_name, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (256, 256))
        image = cv2.bilateralFilter(image, 5, 25, 25)
        
        # Convert the NumPy array to a PIL Image
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if self.transform is not None:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.dataframe.iloc[idx]['Feature'])
            label = torch.tensor(label, dtype=torch.long)
            return image, label

# Transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CSVs
print("DEBUG starting model")
train_df = pd.read_csv('../../../data/student_labels/train2023.csv')
test_df = pd.read_csv('../../../data/student_labels/solution_ids.csv')


def create_feature_df(df, feature):
    df = df[df['Path'].str.startswith('t')].copy()
    df.fillna(0, inplace=True)
    df = df[['Path', feature]].reset_index(drop=True)
    df.rename(columns={feature: 'Feature'}, inplace=True)
    return df

feature_df = create_feature_df(train_df, feature)
train_dataset = ImageDataset(dataframe=feature_df, root_dir='../../../data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

print("DEBUG data loaded")

# Training Function
def train_nn(model, train_loader):
    # Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):  # Number of epochs
        print(f"DEBUG epoch: {epoch}")
        i = 0
        for data, target in train_loader:
            data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32).unsqueeze(1)
            optimizer.zero_grad()     # Zero the gradients
            output = model(data)      # Forward pass
            loss = loss_fn(output, target)  # Compute loss
            loss.backward()           # Backward pass
            optimizer.step()          # Update weights
    return model


# Function to get predictions
def get_output(train_loader, test_loader):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    
    model = model.to(device)

    train_nn(model, train_loader)


    model.eval()
    test_preds = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device, dtype=torch.float32)
            output = model(data)
            output = output.cpu()
            test_preds.extend(output.flatten().tolist())

    return test_preds

def create_test_df(test_df):
    test_df = test_df.copy()
    test_df = test_df.reset_index(drop=True)
    return test_df

test_df = create_test_df(test_df)
test_dataset = ImageDataset(dataframe=test_df, root_dir='../../../data', transform=transform, test=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

print("DEBUG test data loaded")

classification_dict = {}
classification_dict["Id"] = list(test_df['Id'])

print(f"DEBUG running {feature}")
classifications = get_output(train_loader, test_loader)
classification_dict[feature] = classifications

    

print(f"DEBUG {feature} exporting data")
submission_df = pd.DataFrame(classification_dict)
submission_df = submission_df.sort_values(by = "Id")

feature_under = feature.replace(" ", "_")
submission_df.to_csv(f'solutionresults/3resnetfinal{feature_under}.csv', index=False)

