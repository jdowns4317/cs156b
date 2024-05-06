import os
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Check CUDA availability and select the appropriate device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

features = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", 
            "Lung Opacity", "Pneumonia", "Pleural Effusion", "Pleural Other",
            "Fracture", "Support Devices"]

bs = 32
num_epochs = 3
w = 30
h = 30
nw = 4

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
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = self.dataframe.iloc[idx]['Feature'] + 1
            label = torch.tensor(label, dtype=torch.long)
            return image, label

# Transformation
transform = transforms.Compose([
    transforms.Resize((w, h)),  # Resize the image
    transforms.ToTensor(),      # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CSVs
train_df = pd.read_csv('../../../data/student_labels/train2023.csv')
test_df = pd.read_csv('../../../data/student_labels/test_ids.csv')

def create_feature_df(df, feature):
    df = df[df['Path'].str.startswith('t')]
    df = df.dropna(subset=[feature]).query("Path.str.contains('frontal')")
    df = df[['Path', feature]].reset_index(drop=True)
    df.rename(columns={feature: 'Feature'}, inplace=True)
    return df

dl_dict = {}
for feature in features:
    feature_df = create_feature_df(train_df, feature)
    train_dataset = ImageDataset(dataframe=feature_df, root_dir='../../../data', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw)
    dl_dict[feature] = train_loader

# Training Function
def train_nn(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
    return model

# Function to get predictions
def get_output(train_loader, test_loader):
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.5),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.5),
        nn.Flatten(),
        nn.Linear(64 * (w//4) * (h//4), 128),
        nn.ReLU(),
        nn.Linear(128, 3)
    ).to(device)

    train_nn(model, train_loader)

    model.eval()
    test_preds = []
    test_probs = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_preds.extend(pred.flatten().tolist())
            
            probabilities = F.softmax(output, dim=1)
            class_labels = torch.tensor([-1, 0, 1], device=probabilities.device)
            expected_values = torch.sum(probabilities * class_labels, dim=1)
            test_probs.extend(expected_values.flatten().tolist())

    return test_preds, test_probs

# Initialize Dataset and DataLoader for testing
test_dataset = ImageDataset(dataframe=test_df, root_dir='../../../data', transform=transform, test=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=nw)

classification_dict = {}
classification_dict["Id"] = test_df['Id']

probs_dict = {}
probs_dict["Id"] = test_df['Id']

for feature in features:
    classification_dict[feature], probs_dict[feature] = get_output(dl_dict[feature], test_loader)
    classification_dict[feature] = [pred - 1 for pred in classification_dict[feature]]

submission_df = pd.DataFrame(classification_dict)
submission_df.to_csv('cnn_gobeavers_submission.csv', index=False)

probs_df = pd.DataFrame(probs_dict)
probs_df.to_csv('cnn_probs_submission.csv', index=False)
