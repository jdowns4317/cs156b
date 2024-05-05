import os
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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
    transforms.ToTensor()         # Convert images to PyTorch tensors
])

# Load CSVs
print("DEBUG starting model")
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
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=nw)  # Adjust num_workers based on your system
    dl_dict[feature] = train_loader

print("DEBUG data loaded")

# Training Function
def train_nn(model, train_loader):
    # Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):  # Number of epochs
        for data, target in train_loader:
            optimizer.zero_grad()     # Zero the gradients
            output = model(data)      # Forward pass
            loss = loss_fn(output, target)  # Compute loss
            loss.backward()           # Backward pass
            optimizer.step()          # Update weights
    return model


# Function to get predictions
def get_output(train_loader, test_loader):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3*w*h, 20),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(20, 3)  # Adjusted output size for multi-label classification
    )

    train_nn(model, train_loader)


    model.eval()
    test_preds = []
    test_probs = []
    with torch.no_grad():
        for data in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_preds.extend(pred.flatten().tolist())
            
            probabilities = F.softmax(logits_adjusted, dim=1)  # converting logits to probabilities
            class_labels = torch.tensor([-1, 0, 1], device=probabilities.device)  # class labels tensor
            # Calculating expected values: sum of (probability * class_label) across each class
            expected_values = torch.sum(probabilities * class_labels, dim=1)  # shape [batch_size]
            test_probs.extend(expected_values.flatten().tolist())


    return test_preds, test_probs

# Initialize Dataset and DataLoader for testing
test_dataset = ImageDataset(dataframe=test_df, root_dir='../../../data', transform=transform, test=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=nw)

print("DEBUG test data loaded")

classification_dict = {}
classification_dict["Id"] = test_df['Id']

probs_dict = {}
probs_dict["Id"] = test_df['Id']

for feature in features:
    print(f"DEBUG running {feature}")
    classification_dict[feature], probs_dict[feature] = get_output(dl_dict[feature], test_loader)
    classification_dict[feature] = [pred - 1 for pred in classification_dict[feature]]

print("DEBUG exporting data")
submission_df = pd.DataFrame(classification_dict)
submission_df.to_csv('gobeavers_submission.csv', index=False)

probs_df = pd.DataFrame(probs_dict)
probs_df.to_csv('probs_submission.csv', index=False)



